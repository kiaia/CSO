#

# convert from smolagent outputs

import sys
import json
import re
from ..gaia_scorer import question_scorer

def get_str(pat: str, str_step: str):
    gs = re.findall(pat, str_step)
    if len(gs) == 0:
        return ""
    ret = gs[0].replace('"""', '""')  # simply the first
    ret = eval(f'""" {ret} """')
    return ret

def main(input_path: str, output_path="", ref_path=""):
    # the task-id orders
    ref_insts = {}
    if ref_path:
        with open(ref_path) as fd:
            for line in fd:
                if line.strip():
                    inst = json.loads(line)
                    ref_insts[inst["task_id"]] = inst
    ref_path = ref_path if ref_path else input_path
    with open(ref_path) as fd:
        ref_ids = [json.loads(line)["task_id"] for line in fd if line.strip()]
    # get all the results
    results = {}
    with open(input_path) as fd:
        for line in fd:
            old_inst = json.loads(line)
            session = {"task": old_inst["question"], "steps": []}
            for old_step in old_inst["intermediate_steps"]:
                if old_step.startswith("PlanningStep"):
                    plan = get_str(r", plan=['\"](.*)['\"]", old_step)
                    new_step = {"plan": {"code": plan}}
                elif old_step.startswith("ActionStep"):
                    code = get_str(r", model_output=['\"](.*)end_code", old_step)
                    observation = get_str(r", observations=['\"](.*)['\"], observations_images", old_step)
                    new_step = {"action": {"code": code, "observation": observation}}
                else:
                    assert any(old_step.startswith(z) for z in ["TaskStep"])
                    continue
                new_step["step_idx"] = len(session["steps"])
                session["steps"].append(new_step)
            new_inst = {
                "id": old_inst["task_id"],
                "task": old_inst["question"],
                "_orig": ref_insts.get(old_inst["task_id"], {"Level": old_inst["task"]}),
                "session": session,
                "eval": {"pred": old_inst["prediction"], "gold": old_inst["true_answer"], "corr": int(question_scorer(model_answer=old_inst["prediction"], ground_truth=old_inst["true_answer"]))},
            }
            results[new_inst["id"]] = new_inst
    # write the results
    if output_path:
        with open(output_path, 'w') as wfd:
            for one_id in ref_ids:
                wfd.write(json.dumps(results[one_id], ensure_ascii=False) + "\n")
    # --

# python -mpdb -m ckv3.ck_main.scripts.convert_from_smolagent
if __name__ == '__main__':
    main(*sys.argv[1:])
