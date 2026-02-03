#

import os
import time
import random
import re
import sys
import json
import types
import contextlib
from typing import Union, Callable
from functools import partial
import signal
import numpy as np

from rich.console import Console as rich_console
from rich import print as rich_print
from rich.markup import escape as rich_escape

# rprint
_console = rich_console(force_terminal=(False if os.getenv("NO_FORCE_TERMINAL", False) else True))
def rprint(inputs, style=None, timed=False):
    if isinstance(inputs, str):
        inputs = [inputs]  # with style as the default
    all_ss = []
    for one_item in inputs:
        if isinstance(one_item, str):
            one_item = (one_item, None)
        one_str, one_style = one_item  # pairs
        one_str = rich_escape(one_str)
        one_style = style if one_style is None else one_style
        if one_style:
            one_str = f"[{one_style}]{one_str}[/]"
        all_ss.append(one_str)
    _to_print = "".join(all_ss)
    if timed:
        _to_print = f"[{time.ctime()}] {_to_print}"
    # rich_print(_to_print)
    _console.print(_to_print)

# --
# simple adpators
zlog = rprint
zwarn = lambda x: rprint(x, style="white on red")
# --

class MyJsonEncoder(json.JSONEncoder):
    def default(self, one: object):
        if hasattr(one, "to_dict"):
            return one.to_dict()
        if isinstance(one, list):
            one = [item.to_dict() if hasattr(item, "to_dict") else item for item in one]
        else:
            try:
                return json.JSONEncoder.default(self, one)
            except:  # note: simply return a str for whatever the python code may return
                zwarn(f"WARNING: MyJsonEncoder cannot encode the object of {type(one)}, simply put its str: {str(one)}")
                return str(one)

def my_json_dumps(*args, **kwargs):
    return json.dumps(*args, **kwargs, cls=MyJsonEncoder)


def prune_traj_image_session(session):
    for main_step in session['steps']:
        action_dict = main_step['action']
        if "observation" in action_dict and action_dict["observation"] != []:
            ob = action_dict["observation"]
            # is a sub-agent
            if type(ob)==dict and "session" in ob and 'steps' in ob['session']:
                sub_agent_session = ob['session']
                new_subagent_steps = []
                for step in sub_agent_session['steps']:
                    # remove the base64 (calling MLLM) messages
                    if not any(type(p["content"])!=str for p in step['action']['llm_input'] + step['plan']['llm_input']):
                        # remove boxed_screenshot in web_state
                        if "web_state_before" in step['action'] and "boxed_screenshot" in step['action']['web_state_before']:
                            step['action']['web_state_before']['boxed_screenshot'] = None
                        new_subagent_steps.append(step)
                sub_agent_session['steps'] = new_subagent_steps
    return session

def tuple_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): tuple_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tuple_keys_to_str(i) for i in d]
    else:
        return d

# wrapping a function and try it multiple times
def wrapped_trying(func, default_return=None, max_times=10, wait_error_names=(), reraise=False):
    # --
    if max_times < 0:
        return func()  # directly no wrap (useful for debugging)!
    # --
    remaining_tryings = max_times
    ret = default_return
    while True:
        try:
            ret = func()
            break  # remember to jump out!!!
        except Exception as e:
            rprint(f"Retry with Error: {e}", style="white on red")
            rand = random.randint(1, 5)
            time.sleep(rand)
            if type(e).__name__ in wait_error_names:
                continue  # simply wait it
            else:
                remaining_tryings -= 1
                if remaining_tryings <= 0:
                    if reraise:
                        raise e
                    else:
                        break
    return ret

# get env variable until hitting a key or returning the default value
def GET_ENV_VAR(*keys: str, df=None):
    for k in keys:
        if k in os.environ:
            return os.getenv(k)
    return df

# get until hit
def get_until_hit(d, keys, df=None):
    for k in keys:
        if k in d:
            return d[k]
    return df

# easier init with kwargs
class KwargsInitializable:
    def __init__(self, _assert_existing=True, _default_init=False, **kwargs):
        updates = {}
        new_updates = {}
        for k, v in kwargs.items():
            if _assert_existing:
                assert hasattr(self, k), f"Attr {k} not existing!"
            v0 = getattr(self, k, None)
            if v0 is not None and isinstance(v0, KwargsInitializable):
                new_val = type(v0)(**v)  # further make a new one!
                updates[k] = f"__new__ {type(new_val)}"
            elif v0 is None:  # simply directly update
                new_val = v
                new_updates[k] = new_val
            else:
                new_val = type(v0)(v)  # conversion
                updates[k] = new_val
            setattr(self, k, new_val)
        if not _default_init:
            rprint(f"Finish init {self}, updates={updates}, new_updates={new_updates}")

# --
# templated string (also allowing conditional prompts)
class TemplatedString:
    def __init__(self, s: Union[str, Callable]):
        self.str = s

    def format(self, **kwargs):
        if isinstance(self.str, str):
            return TemplatedString.eval_fstring(self.str, **kwargs)
        else:  # direct call it!
            return self.str(**kwargs)

    @staticmethod
    def eval_fstring(s: str, _globals=None, _locals=None, **kwargs):
        if _locals is None:
            _inner_locals = {}
        else:
            _inner_locals = _locals.copy()
        _inner_locals.update(kwargs)
        assert '"""' not in s, "Special seq not allowed!"
        ret = eval('f"""'+s+'"""', _globals, _inner_locals)
        return ret

# a simple wrapper class for with expression
class WithWrapper:
    def __init__(self, f_start: Callable = None, f_end: Callable = None, item=None):
        self.f_start = f_start
        self.f_end = f_end
        self.item: object = item

    def __enter__(self):
        if self.f_start is not None:
            self.f_start()
        if self.item is not None and hasattr(self.item, "__enter__"):
            self.item.__enter__()
        # return self if self.item is None else self.item
        return self.item

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item is not None and hasattr(self.item, "__exit__"):
            self.item.__exit__()
        if self.f_end is not None:
            self.f_end()

def my_open_with(fd_or_path, mode='r', empty_std=False, **kwargs):
    if empty_std and fd_or_path == '':
        fd_or_path = sys.stdout if ('w' in mode) else sys.stdin
    if isinstance(fd_or_path, str) and fd_or_path:
        return open(fd_or_path, mode=mode, **kwargs)
    else:
        # assert isinstance(fd_or_path, IO)
        return WithWrapper(None, None, fd_or_path)

# get unique ID
def get_unique_id(prefix=""):
    import datetime
    import threading
    dt = datetime.datetime.now().isoformat()
    ret = f"{prefix}{dt}_P{os.getpid()}_T{threading.get_native_id()}"  # PID+TID
    return ret

# update dict (in an incremental way)
def incr_update_dict(trg, src_dict):
    for name, value in src_dict.items():
        path = name.split(".")
        curr = trg
        for _piece in path[:-1]:
            if _piece not in curr:  # create one if not existing
                curr[_piece] = {}
            curr = curr[_piece]
        _piece = path[-1]
        if _piece in curr and curr[_piece] is not None:
            assigning_value = type(curr[_piece])(value)  # value to assign
            if isinstance(assigning_value, dict) and isinstance(curr[_piece], dict):
                incr_update_dict(curr[_piece], assigning_value)  # further do incr
            else:
                curr[_piece] = assigning_value  # with type conversion
        else:
            curr[_piece] = value  # directly assign!

# --
# common response format; note: let each agent specify their own ...
# RESPONSE_FORMAT_REQUIREMENT = """## Output
# Please generate your response, your reply should strictly follow the format:
# Thought: {First, explain your reasoning for your outputs in one line.}
# Code: {Then, output your python code blob.}
# """

# parse specific formats
def parse_response(s: str, seps: list, strip=True, return_dict=False):
    assert len(seps) == len(set(seps)), f"Repeated items in seps: {seps}"
    ret = []
    remaining_s = s
    # parse them one by one
    for one_sep_idx, one_sep in enumerate(seps):
        try:
            p1, p2 = remaining_s.split(one_sep, 1)
            if p1.strip():
                rprint(f"Get an unexpected piece: {p1}")
            sep_val = p2
            for one_sep2 in seps[one_sep_idx+1:]:
                if one_sep2 in p2:
                    sep_val = p2.split(one_sep2, 1)[0]
                    break  # finding one is enough!
            assert p2.startswith(sep_val), "Internal error for unmatched prefix??"
            remaining_s = p2[len(sep_val):]
            one_val = sep_val
        except:  # by default None
            one_val = None
        ret.append(one_val)
    # --
    if strip:
        if isinstance(strip, str):
            ret = [(z.strip(strip) if isinstance(z, str) else z) for z in ret]
        else:
            ret = [(z.strip() if isinstance(z, str) else z) for z in ret]
    if return_dict:
        ret = {k: v for k, v in zip(seps, ret)}
    return ret

class CodeExecutor:
    def __init__(self, global_dict=None):
        # self.code = code
        self.results = []
        self.globals = global_dict if global_dict else {}
        # self.additional_imports = None
        self.internal_functions = {"print": self.custom_print, "input": CodeExecutor.custom_input, "exit": CodeExecutor.custom_exit}  # customized ones
        self.null_stdin = not bool(int(GET_ENV_VAR("NO_NULL_STDIN", df="0")))  # for easier debugging and program interacting

    def add_global_vars(self, **kwargs):
        self.globals.update(kwargs)

    @staticmethod
    def extract_code(s: str):
        # CODE_PATTERN = r"```(?:py[^t]|python)(.*?)```"
        CODE_PATTERN = r"```(?:py[^t]|python)(.*)```"  # get more codes
        orig_s, hit_code = s, False
        # strip _CODE_PREFIX
        _CODE_PREFIX = "<|python_tag|>"
        if _CODE_PREFIX in s:  # strip _CODE_PREFIX
            hit_code = True
            _idx = s.index(_CODE_PREFIX)
            s = s[_idx+len(_CODE_PREFIX):].lstrip()  # strip tag
        # strip all ```python ... ``` pieces
        # m = re.search(r"```python(.*)```", s, flags=re.DOTALL)
        if "```" in s:
            hit_code = True
            all_pieces = []
            for piece in re.findall(CODE_PATTERN, s, flags=re.DOTALL):
                all_pieces.append(piece.strip())
            s = "\n".join(all_pieces)
        # --
        # cleaning
        while s.endswith("```"):  # a simple fix
            s = s[:-3].strip()
        ret = (s if hit_code else "")
        return ret

    def custom_print(self, *args):
        # output = " ".join(str(arg) for arg in args)
        # results.append(output)
        self.results.extend(args)  # note: simply adding!

    @staticmethod
    def custom_input(*args):
        return "No input available."

    @staticmethod
    def custom_exit(*args):
        return "Cannot exit."

    def get_print_results(self, return_str=False, clear=True):
        ret = self.results.copy()  # a list of results
        if clear:
            self.results.clear()
        if len(ret) == 1:
            ret = ret[0]  # if there is only one output
        if return_str:
            ret = "\n".join(ret)
        return ret

    def _exec(self, code, null_stdin, timeout):
        original_stdin = sys.stdin  # original stdin
        if timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        try:
            with open(os.devnull, 'r') as fd:
                if null_stdin:  # change stdin
                    sys.stdin = fd
                exec(code, self.globals)  # note: no locals since things can be strange!
        finally:
            if null_stdin:  # change stdin
                sys.stdin = original_stdin
            if timeout > 0:
                signal.alarm(0)  # Disable the alarm
            # simply remove global vars to avoid pickle errors for multiprocessing running!
            # self.globals.clear()  # note: simply create a new executor for each run!

    def run(self, code, catch_exception=True, null_stdin=None, timeout=0):
        if null_stdin is None:
            null_stdin = self.null_stdin  # use the default one
        # --
        if code:  # some simple modifications
            code_nopes = []
            code_lines = [f"import {lib}\n" for lib in ["os", "sys"]] + ["", ""]
            for one_line in code.split("\n"):
                if any(re.match(r"from\s*.*\s*import\s*"+function_name, one_line.strip()) for function_name in self.globals.keys()):  # no need of such imports
                    code_nopes.append(one_line)
                else:
                    code_lines.append(one_line)
            code = "\n".join(code_lines)
            if code_nopes:
                zwarn(f"Remove unneeded lines of {code_nopes}")
        self.globals.update(self.internal_functions)  # add internal functions
        # --
        if catch_exception:
            try:
                self._exec(code, null_stdin, timeout)
            except Exception as e:
                err = self.format_error(code)
                # self.results.append(err)
                if self.results:
                    err = f"{err.strip()}\n(* Partial Results={self.get_print_results()})"
                if isinstance(e, TimeoutError):
                    err = f"{err}\n-> Please revise your code and simplify the next step to control the runtime."
                self.custom_print(err)  # put err
                zwarn(f"Error executing code: {e}")
        else:
            self._exec(code, null_stdin, timeout)
        # --

    @staticmethod
    def format_error(code: str):
        import traceback
        err = traceback.format_exc()
        _err_line = None
        _line_num = None
        for _line in reversed(err.split("\n")):
            ps = re.findall(r"line (\d+),", _line)
            if ps:
                _err_line, _line_num = _line, ps[0]
                break
        # print(_line_num, code.split('\n'))
        try:
            _line_str = code.split('\n')[int(_line_num)-1]
            err = err.replace(_err_line, f"{_err_line}\n    {_line_str.strip()}")
        except:  # if we cannot get the line
            pass
        return f"Code Execution Error:\n{err}"

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded timeout")

def get_np_generator(seed):
    return np.random.RandomState(seed)

# there are images in the messages
def have_images_in_messages(messages):
    for message in messages:
        contents = message.get("content", "")
        if not isinstance(contents, list):
            contents = [contents]
        for one_content in contents:
            if isinstance(one_content, dict):
                if one_content.get("type") == "image_url":
                    return True
    return False
