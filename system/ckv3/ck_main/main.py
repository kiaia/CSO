#

import os
import argparse
import json
import shutil
import time
import multiprocessing as mp
import signal

from ..agents.utils import rprint, my_open_with, zwarn, incr_update_dict, get_until_hit, my_json_dumps, prune_traj_image_session, tuple_keys_to_str
from ..agents.evaluator import Evaluator, question_scorer_llm

from .agent import CKAgent
from .process_reward_agent import ProcessRewardCKAgent
from .gaia_scorer import question_scorer



default_main_configs = {
    "model": {"call_target": "gpt:gpt-4o-mini"},  # LLM target
    "web_agent": {
        "model": {"call_target": "gpt:gpt-4o-mini"},  # LLM target for the web agent
        "web_env_kwargs": {"web_ip": "localhost:3000"},  # IP for the web-browser server
    },
    "process_reward": {
        "enable_process_reward": False,  # Enable Process Reward Model
        "num_action_candidates": 3,     # Number of action candidates to generate per step
        "enable_parallel_sampling": True,  # Enable parallel candidate generation
        "sampling_temperature": 0.8,   # [DEPRECATED] No longer used - only seed variation for diversity
        "min_score_threshold": 0.3,    # Minimum acceptable RPM score
        "prm_usage_strategy": "consistent",  # Strategy: consistent, selective, always
        "rpm_model_config": {
            "call_target": "claude:claude-3.7",  # RPM model endpoint
            "enable_parallel_evaluation": True,   # Enable parallel RPM evaluation
            "evaluation_timeout": 30,             # Timeout for RPM calls
        }
    }
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="")
    parser.add_argument("-u", "--updates", type=str, default=[], nargs="+")  # updating dicts
    parser.add_argument("-i", "--input", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="")
    parser.add_argument("-S", "--sep_run_root", type=str, default="")  # separate running root dir for each task, empty if not enabled
    parser.add_argument("-P", "--try_preload_output", type=int, default=1)  # try loading the already processed outputs
    parser.add_argument("--starting_idx", type=int, default=0)  # starting with which one?
    parser.add_argument("--no_final_breakpoint", type=str, default="")
    parser.add_argument("--skip-hard-query", action="store_true")
    parser.add_argument("--reomve-image-base64-cache", action="store_true")
    parser.add_argument("--sampling-mode", action="store_true") # in sampling (trajectory) model, will use an evaluator to check whether the query has finished (with an answer)
    parser.add_argument("--sampling-evaluation-method", choices=["disabled", "em", "llm_score", "stop_with_answer"], default="disabled") # useful when --sampling-mode is on.
    # types of auto evaluator. em: exact match; llm_score: llm score using langchain; Answers should be provided in this mode. stop_with_answer: simply determine whether the query stops with an answer. suits the situation where no ground answers are provided.
    parser.add_argument("--inference-time-evaluation-method", choices=["disabled", "no_answer", "no_answer+no_ask_llm", "gpt_judge", "ensemble", "gpt_judge+ensemble"], default="disabled") # whether to enable an auto evaluator and perform reflection
    parser.add_argument("--max_retry_num", default=3, type=int) # maximum number of retries when sampling-mode or inference_time_evaluation_method is on.
    parser.add_argument("--reflection", type=bool, default=False)
    parser.add_argument("--evaluation-metric", choices=["em", "llm_score", "llm_score_prompt"], default="em") # em: GAIA scorer based on exact-match. llm_score: using llm-based similarity match
    parser.add_argument("--save_failed_tries", action="store_true") # whether to save "failed" tries. Can disable this when running inference on test set.
    # parser.add_argument("-t", "--timeout", type=int, default=3600)  # timeout seconds for each task
    return parser.parse_args()

def yield_inputs(input_file):
    _idx = 0
    if input_file:
        with open(input_file) as fd:
            for line in fd:
                if line.strip():
                    one_inst = json.loads(line)
                    # if there is info
                    info_field = one_inst['info'] if 'info' in one_inst else one_inst
                    # get fields
                    task = get_until_hit(info_field, ["question", "Question", "task", "Task", "query", "Query", "instruction", "Instruction"])
                    file = get_until_hit(info_field, ["file_name"])
                    answer = get_until_hit(info_field, ["Final answer", "answer", "true_answer"])
                    if get_until_hit(info_field, ["skip"]) is None or str(get_until_hit(info_field, ["skip"])) != '1':
                        skip_hard = False
                    else:
                        skip_hard = True
                    if task:
                        yield {"id": f"task{_idx:04d}", "task": task, "file": file, "answer": answer, "_orig": one_inst, "skip_hard": skip_hard}
                        _idx += 1
                    else:
                        zwarn(f"Cannot find task from: {one_inst}")
    else:  # read from input
        while True:
            task = input("Input your task prompt >> ").strip()
            if not task:
                continue
            if task == "__END__":
                break
            yield {"id": f"task{_idx:04d}", "task": task}
            _idx += 1

# --
def main():
    args = get_args()
    rprint(f"Run ck_main.main with {args}")
    mp.set_start_method("spawn")
    # signal.signal(signal.SIGALRM, timeout_handler)
    # --
    # init agent
    configs = default_main_configs
    if args.config:
        with open(args.config) as fd:
            configs = json.load(fd)
        rprint(f"Load configs from {args.config} = {configs}")
    for one_update in args.updates:
        src_dict = eval(one_update)
        incr_update_dict(configs, src_dict)  # updates
        rprint(f"Update configs with {src_dict}")
    
    # Choose agent type based on process reward configuration
    process_reward_config = configs.get("process_reward", {})
    if process_reward_config.get("enable_process_reward", False):
        rprint("Initializing ProcessRewardCKAgent")
        ck_agent = ProcessRewardCKAgent(**configs)
    else:
        rprint("Initializing standard CKAgent")
        # Remove process_reward config for standard CKAgent to avoid attribute error
        ck_configs = {k: v for k, v in configs.items() if k != "process_reward"}
        ck_agent = CKAgent(**ck_configs)
    if args.sampling_mode or args.inference_time_evaluation_method != "disabled" or args.evaluation_metric == "llm_score":
        ck_evaluator = Evaluator()
    # --
    old_dir = os.path.abspath(os.curdir)
    input_dir = os.getenv("FILE_BASE_DIR", default=os.path.dirname(os.path.abspath(args.input)))
    if args.sep_run_root:  # mkdir
        os.makedirs(args.sep_run_root, exist_ok=True)
    # --
    existing_inst_map = {}
    if args.try_preload_output and os.path.exists(args.output):
        with open(args.output) as fd:
            for line in fd:
                if line.strip():
                    _inst = json.loads(line)
                    existing_inst_map[_inst["id"]] = _inst
    if existing_inst_map:
        rprint(f"Load existing_inst_map: L={len(existing_inst_map)}")
    # --
    total_task, corr_task = 0, 0
    with my_open_with(args.output, 'w') as fout:
        for inst in yield_inputs(args.input):
            _input_file = None
            if args.sep_run_root:
                trg_dir = os.path.join(args.sep_run_root, inst["id"])
                os.makedirs(trg_dir, exist_ok=False)  # mkdir
                if inst.get("file"):
                    _input_file = "input." + inst["file"].split(".")[-1]  # make a simpler name!
                    shutil.copy(os.path.join(input_dir, inst["file"]), os.path.join(trg_dir, _input_file))
                os.chdir(trg_dir)  # switch to specific working dir
            else:
                _input_file = os.path.join(input_dir, inst["file"]) if inst.get("file") else None
            _task = inst["task"].strip()
            if _input_file:
                _task = f"{_task}\n(* You are given the following input file: {_input_file})"
            rprint(f"Start to run task {inst['id']}", timed=True)
            
            # breakpoint()
            if inst["id"] in existing_inst_map:  # simply load it
                exist_inst = existing_inst_map[inst["id"]]
                if exist_inst["task"] != inst["task"]:
                    zwarn(f"Ignore mismatched instances: {exist_inst['task']} vs {inst['task']}")
                else:
                    rprint(f"Directly load the previous run session without running for {inst['id']}")
                    if "session" in exist_inst:
                        inst["session"] = exist_inst["session"]
                    else:
                        inst["session"] = ""
                        # pass
            else:
                # queries that have not been processed. 
                # if there's a skip key in the file, simply skip the hard query.
                # print(inst['id'],  inst['skip_hard'])
                if args.skip_hard_query and inst['skip_hard']:
                    if fout:
                        inst['eval'] =  {"pred": 'NA', "gold": str(inst.get("answer", "UNK")), "corr": 0}
                        inst['session'] = {}
                        fout.write(my_json_dumps(inst, ensure_ascii=False) + "\n")
                        continue
            
            res_session_list = []
            if "session" not in inst:
                start_pc, start_time = time.perf_counter(), time.ctime()
                if total_task >= args.starting_idx:
                    if args.sampling_mode:
                        # sampling mode
                        if args.sampling_evaluation_method == "disabled":
                            res_session = ck_agent.run(_task)
                        elif args.sampling_evaluation_method in ["em", "llm_score"]:
                            _try_num = 0
                            res_session = ck_agent.run(_task)
                            res_session_list.append(res_session)
                            while _try_num < args.max_retry_num:
                                _try_num += 1
                                if ck_evaluator.evaluate_with_answer(res_session.to_dict(), str(inst.get("answer", "UNK")), inst["task"].strip(), evaluation_method=args.sampling_evaluation_method):
                                    res_session = ck_agent.run(_task)
                                    res_session_list.append(res_session)
                                else:
                                    break
                    else:
                        # inference mode
                        if args.inference_time_evaluation_method == "disabled":
                            res_session = ck_agent.run(_task)
                        else:
                            # ensemble
                            candidate_num = 5 if "ensemble" in args.inference_time_evaluation_method else 1
                            candidate_sessions = []
                            # retry
                            for i in range(candidate_num):
                                rprint(f"Start to run task {inst['id']} for the {i+1} time", timed=True)
                                feedback = None
                                feedback_list = []
                                for j in range(args.max_retry_num):
                                    if args.reflection:
                                        new_task = f"{_task}. Here is a feedback for a previous try that failed:\n\n{feedback}" if feedback else _task
                                    else:
                                        new_task = _task
                                    res_session = ck_agent.run(new_task)
                                    res_session_list.append(res_session)
                                    has_failure, feedback = ck_evaluator.detect_failure(res_session.to_dict(), evaluation_type=args.inference_time_evaluation_method)
                                    if not has_failure:
                                        break
                                    print(f"Retrying task {inst['id']} due to {feedback}")
                                    feedback_list.append(feedback)
                                candidate_sessions.append(res_session)
                            if "ensemble" in args.inference_time_evaluation_method:
                                res_session = candidate_sessions[ck_evaluator.ensemble([x.to_dict() for x in candidate_sessions])]
                            inst["feedback"] = feedback_list
                else:
                    res_session = None
                    rprint(f"Skipping task {inst['id']}")
                if res_session is None:  # error?
                    inst["session"] = {"steps": [{"step_idx": -1, "end": {"final_results": {"output": "error", "log": "error"}}}]}
                else:
                    res_session.info["call_stat"] = ck_agent.get_call_stat(clear=True)
                    end_pc, end_time = time.perf_counter(), time.ctime()
                    res_session.info.update({"start_time": start_time, "end_time": end_time, "duration": end_pc-start_pc})
                    inst["session"] = res_session.to_dict()
                    if args.save_failed_tries and len(res_session_list) > 1:
                        inst['previous_failed_sessions'] = [sess.to_dict() for sess in res_session_list[:-1]]
            # --
            # simple EVAL
            answer_gold = str(inst.get("answer", "UNK"))
            try:
                answer_pred = str(inst["session"]["steps"][-1]["end"]["final_results"]["output"])
            except:
                answer_pred = "error"
            total_task += 1
            if args.evaluation_metric == "em":
                _this_corr = int(question_scorer(model_answer=answer_pred, ground_truth=answer_gold))
            elif args.evaluation_metric == "llm_score_prompt":
                _this_corr = int(question_scorer_llm(answer_pred, answer_gold, _task))
            elif args.evaluation_metric == "llm_score":
                _this_corr = int(ck_evaluator.cot_qa_evaluate({"pred":answer_pred, "gold":answer_gold, "task":_task}))
            else:
                raise NotImplementedError
            corr_task += _this_corr
            inst["eval"] = {"pred": answer_pred, "gold": answer_gold, "corr": _this_corr}  # store the eval results
            rprint(f"Evaluating pred={answer_pred} vs gold={answer_gold}")
            rprint(f"Current Processing Accuracy = {corr_task}/{total_task}={corr_task/total_task:.4f}")
            # =====
            # save
            if args.sep_run_root:
                os.chdir(old_dir)  # switch back
            if fout:
                try:
                    # use my_json_dumps to iteratively convert sub-agent session to dict()
                    _dump = my_json_dumps(tuple_keys_to_str(inst), ensure_ascii=False)
                    if args.reomve_image_base64_cache:
                        _inst = json.loads(_dump)
                        _inst['session'] = prune_traj_image_session(_inst['session'])
                        _dump = json.dumps(_inst, ensure_ascii=False)
                    fout.write(_dump + "\n")
                except:
                    print("error writing instance")
                    inst = dict([(key, inst[key]) for key in ['id', 'task', 'file', 'answer', '_orig', 'skip_hard', 'eval']])
                    _dump = my_json_dumps(tuple_keys_to_str(inst), ensure_ascii=False)
                    if args.reomve_image_base64_cache:
                        _inst = json.loads(_dump)
                        _inst['session'] = prune_traj_image_session(_inst['session'])
                        _dump = json.dumps(_inst, ensure_ascii=False)
                    fout.write(_dump + "\n")
                    # breakpoint()
            # --
    # --
    if args.no_final_breakpoint:
        pass
    else:
        rprint("Yeah, everything has been finished!!!!!")
        # breakpoint()
    # --

# --
# python -m ckv3.ck_main.main --config config.json --input ... --output ... --sep_run_root test0
if __name__ == '__main__':
    main()
