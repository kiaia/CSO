#

import time
import argparse
import json
from .agent import WebAgent
from ..agents.utils import rprint, my_open_with, zwarn, incr_update_dict, get_until_hit, my_json_dumps

default_web_configs = {
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
    parser.add_argument("-P", "--try_preload_output", type=int, default=1)  # try loading the already processed outputs
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
                    target_url = get_until_hit(info_field, ["target_url"], df=DEFAULT_START_PAGE)
                    if task:
                        yield {"id": f"task{_idx:04d}", "task": task, "target_url": target_url, "_orig": one_inst}
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
            _special_prefix = "[[[[[http"
            if _special_prefix in task:
                _idx = task.index(_special_prefix)
                task, target_url = task[:_idx], task[_idx+len(_special_prefix)-4:]
            else:
                target_url = DEFAULT_START_PAGE
            yield {"task": task, "target_url": target_url}


import subprocess
import os
import signal

def kill_web():
    """Kill the web environment started with 'sh run_local_mac.sh'."""
    try:
        # Find the process ID (PID) of the running web service
        # This assumes that the script is named 'run_web.sh'
        # and that it is running in the background.
        # You may need to adjust the command based on how the script is run.
        result = subprocess.run(
            ["pgrep", "-f", "run_local_mac.sh"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # If the command was successful, we have the PID(s)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)  # Send SIGTERM to the process
            print("Web environment killed successfully.")
        else:
            print("No running web environment found.")

        port_result = subprocess.run(
            ["lsof", "-t", "-i:3000"],
            capture_output=True,
            text=True
        )
        
        if port_result.returncode == 0:
            # If the command was successful, we have the PID(s) using port 3000
            port_pids = port_result.stdout.strip().split('\n')
            for port_pid in port_pids:
                os.kill(int(port_pid), signal.SIGTERM)  # Send SIGTERM to the process
            print("Process using port 3000 killed successfully.")
        else:
            print("No process found using port 3000.")
    
    except Exception as e:
        print(f"An error occurred while trying to kill the web environment: {e}")

def start_web():
    """Start the web environment using 'sh run_web.sh'."""
    try:
        # Start the web service
        # script_directory = "/path/to/ckv3/ck_web/_web"
        script_directory = f"{os.getcwd()}/ckv3/ck_web/_web"
        subprocess.Popen(["sh", "run_local_mac.sh"], cwd=script_directory)
        print("Web environment started successfully.")
    except Exception as e:
        print(f"An error occurred while trying to start the web environment: {e}")

# --
def main():
    args = get_args()
    rprint(f"Run ck_web.main with {args}")
    # --
    # init agent
    configs = default_web_configs
    if args.config:
        with open(args.config) as fd:
            configs = json.load(fd)
        rprint(f"Load configs from {args.config} = {configs}")
    if args.updates:
        src_dict = eval(args.updates)
        incr_update_dict(configs, src_dict)  # updates
        rprint(f"Update configs with {src_dict}")
    web_agent = WebAgent(**configs)
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

    with my_open_with(args.output, 'w') as fout:
        for inst in yield_inputs(args.input):
            if inst["id"] in existing_inst_map:  # simply load it
                exist_inst = existing_inst_map[inst["id"]]
                if exist_inst["task"] != inst["task"]:
                    zwarn(f"Ignore mismatched instances: {exist_inst['task']} vs {inst['task']}")
                else:
                    rprint(f"Directly load the previous run session without running for {inst['id']}")
                    inst["session"] = exist_inst["session"]
            if "session" not in inst:
                res_session = web_agent.run(inst["task"], target_url=inst["target_url"])
                inst["session"] = res_session.to_dict()
            if fout:
                fout.write(my_json_dumps(inst, ensure_ascii=False) + "\n")
    # --

# --
# python -m ckv3.ck_web.main --config config.json --input ... --output ...
if __name__ == '__main__':
    main()
