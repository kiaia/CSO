#

# some analysis on the output file

import json
import sys
import argparse
from collections import Counter, defaultdict
from ...agents.utils import rprint

def print_session(session):
    all_sub_sessions = {}
    colors = {"plan": "white on blue", "action": "white on green", "end": "white on red"}
    for step_info in session["steps"]:
        step_idx = step_info["step_idx"]
        rprint(f"# ===== Step {step_idx}")
        for key in ["plan", "action", "end"]:
            if key in step_info:
                _to_print = step_info[key].get("llm_output", step_info[key].get("code", ""))
                rprint(str(_to_print), style=colors[key])
                if key == "action":
                    all_obs = step_info[key].get("observation", [])
                    if not isinstance(all_obs, list):
                        all_obs = [all_obs]
                    _printings = []
                    for _obs in all_obs:
                        if isinstance(_obs, dict) and "output" in _obs:
                            if "session" in _obs:
                                # print_session(all_sub_sessions[0])
                                all_sub_sessions[step_idx] = _obs["session"]
                            if "sel_cands" in _obs:
                                # print_session(all_sub_sessions[0][0]['session'])
                                all_sub_sessions[step_idx] = _obs["sel_cands"]
                            _obs = {k: _obs.get(k) for k in ["output", "log", "repr", "sel_idx"]}
                        _printings.append(_obs)
                    if len(_printings) == 1:
                        _printings = _printings[0]
                    rprint(str(_printings), style="white on purple")
    return all_sub_sessions

def analyze(file: str, args):
    cc = Counter()
    token_cc = defaultdict(Counter)
    func_bd = eval(args.breakdowns) if args.breakdowns else (lambda x: None)
    bd_counts, bd_corr = Counter(), Counter()
    with open(file) as fd:
        for line in fd:
            if line.strip():
                inst = json.loads(line)
                _level = inst["_orig"]["Level"]
                _corr = int(inst["eval"]["corr"])  # eval
                rprint(f"Read one inst {inst['id']} (L={_level}): {inst['eval']}")
                cc['inst_all'] += 1
                cc['inst_all_corr'] += _corr
                cc[f'inst_L{_level}'] += 1
                cc[f'inst_L{_level}_corr'] += _corr
                _bd = func_bd(inst)
                bd_counts[_bd] += 1
                bd_corr[_bd] += _corr
                for _kk, _vv in inst.get("session", {}).get("info", {}).get("call_stat", {}).items():
                    if "__ALL__" in _vv:
                        _vv = _vv["__ALL__"]
                    token_cc[_kk].update(_vv)
                # --
                if args.print:
                    if cc['inst_all'] >= args.print_start and ((not args.print_levels) or (int(_level) in args.print_levels)):
                        all_sub_sessions = print_session(inst["session"])
                        rprint(f"# ==\nTask (L={_level}) => {inst['task']}", style="white on yellow")
                        rprint(f"Eval result = {inst['eval']}", style="white on yellow")
                        # if args.breakpoint:
                        #     breakpoint()
                # --
    # --
    acc_results = {}
    for kk in cc.keys():
        for kk2 in cc.keys():
            if kk != kk2 and kk.startswith(kk2):
                acc_results[kk] = cc[kk] / cc[kk2]
    rprint(f"CC for {file}: {cc}")
    rprint(f"Token-CC for {file}: {token_cc}")
    rprint(f"Acc for {file}: {acc_results}")
    if args.breakdowns:
        rprint(f"Breakdown by {args.breakdowns}")
        for kk in sorted(bd_counts.keys()):
            rprint(f"- {kk}: {bd_corr[kk]} / {bd_counts[kk]} = {bd_corr[kk]/bd_counts[kk]}")
    # --

# --
# extra helper function
# present_gaia_dev("../data/gaia_dev.jsonl", "gdev.xlsx")
# Final counts = Counter({'web': 122, 'search': 112, 'calculator': 43, 'image': 30, 'none': 18, 'file.pdf': 16, 'file.table': 13, 'file': 13, 'video': 9, 'audio': 6}) // ALL=165
# Final counts2 = Counter({'web': 122, 'file': 39, 'image': 35, 'none': 18})
def present_gaia_dev(input_file, output_file=""):
    import json
    import pandas as pd
    from collections import Counter
    # --
    _MAPS = {"web": "web", "browser": "web", "wikipedia": "web", "websites": "web", "search": "search", "calculator": "calculator",
             "image": "image", "images": "image", "vision": "image", "ocr": "image", "pdf": "file.pdf", "file": "file", "powerpoint": "file",
             "video": "video", "youtube": "video", "excel": "file.table", "spreadsheet": "file.table", "audio": "audio", "speech-to-text": "audio"}
    _counts = Counter()
    _counts2 = Counter()  # larger category
    all_new_insts = []
    with open(input_file) as fd:
        insts = [json.loads(line) for line in fd]
        for inst in insts:
            new_inst = {k: inst[k] for k in ["Question", "Level", "Final answer"]}
            new_tools = []
            one_tools = [z.split(None, 1)[-1].replace("(Optional)", "").replace(".", "").replace("/", "").strip().lower() for z in inst['Annotator Metadata']['Tools'].split('\n')]
            for one_tool in one_tools:
                matched_kk = None
                for kk1, kk2 in _MAPS.items():
                    if kk1 in one_tool.split():
                        matched_kk = kk2
                        break
                if matched_kk is None:
                    print(f"UNK tool of {one_tool}")
                else:
                    new_tools.append(matched_kk)
            new_inst["Tools"] = ",".join(new_tools) if new_tools else "none"
            all_new_insts.append(new_inst)
            # --
            for kk in new_inst["Tools"].split(","):
                _counts[kk] += 1
            for kk, vv in {"web": ["web", "search"], "file": ["file"], "image": ["image", "video"], "none": ["none"]}.items():
                if any(vv2 in new_inst["Tools"] for vv2 in vv):
                    _counts2[kk] += 1
            # --
    df = pd.DataFrame.from_records(all_new_insts)
    print(f"Final counts = {_counts}")
    print(f"Final counts2 = {_counts2}")
    if output_file:
        df.to_excel(output_file)
    # breakpoint()
# --

# --
def download_gaia():  # nope, need AUTH ...
    import os
    for dd in ["validation", "test"]:
        dd2 = {'validation': 'dev'}.get(dd, dd)
        os.system(f"wget https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/{dd}/metadata.jsonl -O gaia_{dd2}.jsonl")
        with open(f"gaia_{dd2}.jsonl") as fd:
            for line in fd:
                inst = json.loads(line)
                if inst['file_name']:
                    os.system(f"wget https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/{dd}/{inst['file_name']}")

def download_gaia2():
    import datasets
    import shutil
    import json
    z = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")
    for dd in ["validation", "test"]:
        dd2 = {'validation': 'dev'}.get(dd, dd)
        with open(f"gaia_{dd2}.jsonl", "w") as fd:
            for inst in z[dd]:
                fd.write(json.dumps(inst, ensure_ascii=False) + "\n")
                if inst['file_name']:
                    shutil.copy(inst['file_path'], inst['file_name'])
# --

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", type=str, nargs="+", default=[])  # input files to analyze
    parser.add_argument("-p", "--print", type=int, default=1)  # print sessions
    parser.add_argument("-b", "--breakpoint", type=int, default=0)  # add breakpoint
    parser.add_argument("--breakdowns", type=str, default="")  # breaking down function
    parser.add_argument("--print_start", type=int, default=0)
    parser.add_argument("--print_levels", type=int, default=None, nargs="+")
    return parser.parse_args()

def main():
    args = get_args()
    rprint(f"Run analysis with {args}")
    for file in args.files:
        analyze(file, args)

# python -m ckv3.ck_main.scripts.analyze -f ... --breakdowns 'lambda x: f"""L{x["_orig"]["Level"]},F{x["_orig"]["file_name"].split(".")[-1]}"""'
if __name__ == '__main__':
    main()
