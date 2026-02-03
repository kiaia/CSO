#

import os
import time
import argparse
import json
from .agent import FileAgent
from ..agents.utils import rprint, my_open_with, zwarn, incr_update_dict, get_until_hit, my_json_dumps

default_file_configs = {
    "model": {"call_target": "gpt:gpt-4o-mini"},  # LLM target
    # "web_env_kwargs": {"web_ip": "localhost:3000"},  # IP for the web-browser server
}

DEFAULT_START_PAGE = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="")
    parser.add_argument("-u", "--updates", type=str, default="{}")  # an updating dict
    parser.add_argument("-i", "--input", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="")
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
                    
                    file_path_dict = get_until_hit(info_field, ["file_path_dict", "file_name"], df=DEFAULT_START_PAGE)

                    if task:
                        if isinstance(file_path_dict, str): 
                            file_path_dict = {file_path_dict: ""}
                        elif isinstance(file_path_dict, dict): 
                            pass
                        elif isinstance(file_path_dict, list): 
                            file_base_dir = os.getenv("FILE_BASE_DIR", "")
                            file_path_dict = {os.path.join(file_base_dir, file): "" for file in file_path_dict}
                        yield {"task": task, "file_path_dict": file_path_dict, "_orig": one_inst}
                    else:
                        zwarn(f"Cannot find task from: {one_inst}")
    else:  # read from input
        while True:
            task = input("Input your task prompt >> ").strip()
            if not task:
                continue
            if task == "__END__":
                break
            _special_prefix = "[[[[[http"
            if _special_prefix in task:
                _idx = task.index(_special_prefix)
                task, file_path_dict = task[:_idx], task[_idx+len(_special_prefix)-4:]
            else:
                file_path_dict = DEFAULT_START_PAGE
            yield {"task": task, "file_path_dict": file_path_dict}


# --
def main():
    args = get_args()
    rprint(f"Run ck_file.main with {args}")
    # --
    # init agent
    configs = default_file_configs
    if args.config:
        with open(args.config) as fd:
            configs = json.load(fd)
        rprint(f"Load configs from {args.config} = {configs}")
    if args.updates:
        src_dict = eval(args.updates)
        incr_update_dict(configs, src_dict)  # updates
        rprint(f"Update configs with {src_dict}")
    file_agent = FileAgent(**configs)
    # --
    with my_open_with(args.output, 'w') as fout:
        for inst in yield_inputs(args.input):
            res_session = file_agent.run(inst["task"], file_path_dict=inst["file_path_dict"])
            inst["session"] = res_session.to_dict()
            if fout:
                fout.write(my_json_dumps(inst, ensure_ascii=False) + "\n")
    # --

# --
# python -m ckv3.ck_file.main --config config.json --input ... --output ...
if __name__ == '__main__':
    main()
