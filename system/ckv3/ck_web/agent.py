#

import os
import re
import shutil
import urllib.request
from contextlib import contextmanager

from ..agents.agent import MultiStepAgent, register_template, ActionResult
from ..agents.model import LLM
from ..agents.utils import zwarn, rprint, have_images_in_messages
from ..agents.tool import SimpleSearchTool

from .utils import WebEnv
from .prompts import PROMPTS as WEB_PROMPTS

# --
# pre-defined actions: simply convert things to str
def web_click(id: int, link_name=""): return ActionResult(f"click [{id}] {link_name}")
def web_type(id: int, content: str, enter=True): return ActionResult(f"type [{id}] {content}" if enter else f"type [{id}] {content}[NOENTER]")
def web_scroll_up(): return ActionResult(f"scroll up")
def web_scroll_down(): return ActionResult(f"scroll down")
def web_wait(): return ActionResult(f"wait")
def web_goback(): return ActionResult(f"goback")
def web_restart(): return ActionResult(f"restart")
def web_goto(url: str): return ActionResult(f"goto {url}")
# def web_stop(answer, summary): return ActionResult(f"stop [{answer}] ({summary})")  # use self-defined function!
# --

class WebAgent(MultiStepAgent):
    def __init__(self, **kwargs):
        # note: this is a little tricky since things will get re-init again in super().__init__
        feed_kwargs = dict(
            name="web_agent",
            description="A web agent helping to browse and operate web pages to solve a specific task.",
            templates={"plan": "web_plan", "action": "web_action", "end": "web_end"},  # template names
            max_steps=16,
        )
        feed_kwargs.update(kwargs)
        self.web_env_kwargs = {}  # kwargs for web env
        self.check_nodiff_steps = 3  # if for 3 steps, we have the same web page, then explicitly indicating this!
        self.html_md_budget = 0  # budget in bytes (around 4 bytes per token, for example: 2K bytes ~ 500 tokens; 0 means not using this)
        self.use_multimodal = "auto"  # no: always no, yes: always yes, auto: let the agent decide
        self.model_multimodal = LLM(_default_init=True)  # multimodal model
        # self.searcher = SimpleSearchTool(max_results=16, list_enum=False)  # use more!
        # --
        register_template(WEB_PROMPTS)  # add web prompts
        super().__init__(**feed_kwargs)
        self.web_envs = {}  # session_id -> ENV
        self.ACTIVE_FUNCTIONS.update(click=web_click, type=web_type, scroll_up=web_scroll_up, scroll_down=web_scroll_down, wait=web_wait, goback=web_goback, restart=web_restart, goto=web_goto)
        # self.ACTIVE_FUNCTIONS.update(stop=self._my_stop, save=self._my_save, search=self._my_search)
        self.ACTIVE_FUNCTIONS.update(stop=self._my_stop, save=self._my_save, screenshot=self._my_screenshot)
        # --

    # note: a specific stop function!
    def _my_stop(self, answer: str = None, summary: str = None, output: str = None):
        if output:
            ret = f"Final answer: [{output}] ({summary})"
        else:
            ret = f"Final answer: [{answer}] ({summary})"
        self.put_final_result(ret)  # mark end and put final result
        return ActionResult("stop", ret)

    # note: special save
    def _my_save(self, remote_path: str, local_path: str):
        try:
            _dir = os.path.dirname(local_path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            if local_path != remote_path:
                remote_path = remote_path.strip()
                if remote_path.startswith("http://") or remote_path.startswith("https://"):  # retrieve from the web
                    urllib.request.urlretrieve(remote_path, local_path)
                else:  # simply copy!
                    shutil.copyfile(remote_path, local_path)
            ret = f"Save Succeed: from remote_path = {remote_path} to local_path = {local_path}"
        except Exception as e:
            ret = f"Save Failed with {e}: from remote_path = {remote_path} to local_path = {local_path}"
        return ActionResult("save", ret)

    # note: whether use the screenshot mode
    def _my_screenshot(self, flag: bool, save_path=""):
        return ActionResult(f"screenshot {int(flag)} {save_path}")

    def get_function_definition(self, short: bool):
        if short:
            return "- def web_agent(task: str, target_url: str = None) -> Dict:  # Employs a web browser to navigate and interact with web pages to accomplish a specific task. Note that the web agent is limited to downloading files and cannot process or analyze them."
        else:
            return """- web_agent
```python
def web_agent(task: str) -> dict:
    \""" Employs a web browser to navigate and interact with web pages to accomplish a specific task.
    Args:
        task (str): A detailed description of the task to perform. This may include:
            - The target website(s) to visit (include valid URLs).
            - Specific output formatting requirements.
            - Instructions to download files (specify desired output path if needed).
    Returns:
        dict: A dictionary with the following structure:
            {
                'output': <str>  # The well-formatted answer, strictly following any specified output format.
                'log': <str>     # Additional notes, such as steps taken, issues encountered, or relevant context.
            }
    Notes:
        - If the `task` specifies an output format, ensure the 'output' field matches it exactly.
        - The web agent can download files, but cannot process or analyze them. If file analysis is required, save the file to a local path and return control to an external planner or file agent for further processing.
    Example:
        >>> answer = web_agent(task="What is the current club of Messi? (Format your output directly as 'club_name'.)")
        >>> print(answer)  # directly print the full result dictionary
    \"""
```"""

    def __call__(self, task: str, **kwargs):  # allow *args styled calling
        return super().__call__(task, **kwargs)

    def init_run(self, session):
        super().init_run(session)
        _id = session.id
        assert _id not in self.web_envs
        _kwargs = self.web_env_kwargs.copy()
        if session.info.get("target_url"):
            _kwargs["starting_target_url"] = session.info["target_url"]
        self.web_envs[_id] = WebEnv(**_kwargs)

    def end_run(self, session):
        ret = super().end_run(session)
        _id = session.id
        self.web_envs[_id].stop()
        del self.web_envs[_id]  # remove web env
        return ret

    def step_call(self, messages, session, model=None):
        _use_multimodal = session.info.get("use_multimodal", False) or have_images_in_messages(messages)
        if model is None:
            model = self.model_multimodal if _use_multimodal else self.model  # use which model?
        response = model(messages)
        return response

    def step_prepare(self, session, state):
        _input_kwargs, _extra_kwargs = super().step_prepare(session, state)
        _web_env = self.web_envs[session.id]
        _web_state = _web_env.get_state()
        _this_page_info = self._prep_page(_web_state)
        _input_kwargs.update(_this_page_info)  # update for the current one
        if session.num_of_steps() > 1:  # has previous step
            _prev_step = session.get_specific_step(-2)  # the step before
            _input_kwargs.update(self._prep_page(_prev_step["action"]["web_state_before"], suffix="_old"))
        else:
            _input_kwargs["web_page_old"] = "N/A"
        _input_kwargs["html_md"] = self._prep_html_md(_web_state)
        # --
        # check web page differences
        if session.num_of_steps() >= self.check_nodiff_steps and self.check_nodiff_steps > 1:
            _check_pages = [self._prep_page(z["action"]["web_state_before"]) for z in session.get_latest_steps(count=self.check_nodiff_steps-1)] + [_this_page_info]
            if all(z==_check_pages[0] for z in _check_pages):  # error
                _input_kwargs["web_page"] = _input_kwargs["web_page"] + "\n(* Error: Notice that we have been stuck at the same page for many steps, use the `stop` function to terminate and report related errors!!)"
            elif _check_pages[-1] == _check_pages[-2]:  # warning
                _input_kwargs["web_page"] = _input_kwargs["web_page"] + "\n(* Warning: Notice that the web page has not been changed.)"
        # --
        _extra_kwargs["web_env"] = _web_env
        return _input_kwargs, _extra_kwargs

    def step_action(self, action_res, action_input_kwargs, web_env=None, **kwargs):
        action_res["web_state_before"] = web_env.get_state()  # inplace storage of the web-state before the action
        _rr = super().step_action(action_res, action_input_kwargs)  # get action from code execution
        if isinstance(_rr, ActionResult):
            action_str, action_result = _rr.action, _rr.result
        else:
            action_str = self.get_obs_str(None, obs=_rr, add_seq_enum=False)
            action_str, action_result = "nop", action_str.strip()  # no-operation
        # state step
        try:  # execute the action on the browser
            step_result = web_env.step_state(action_str)
            ret = action_result if action_result is not None else step_result  # use action result if there are direct ones
            web_env.sync_files()
            # ret = f"Browser step: {action_str.strip()}"
        except Exception as e:
            zwarn("web_env execution error!")
            ret = f"Browser error: {e}"
        return ret

    # --
    # other helpers

    def _prep_page(self, web_state, suffix=""):
        _ss = web_state
        _ret = _ss["current_accessibility_tree"]
        if _ss["error_message"]:
            _ret = _ret + "\n(Note: " + _ss["error_message"] + ")"
        elif _ss["current_has_cookie_popup"]:
            _ret = _ret + "\n(Note: There is a cookie banner on the page, please accept the cookie banner.)"
        ret = {"web_page": _ret, "downloaded_file_path": _ss["downloaded_file_path"]}
        # --
        if self.use_multimodal == 'on':  # always on
            ret["screenshot"] = _ss["boxed_screenshot"]
        elif self.use_multimodal == 'off':
            ret["screenshot_note"] = "The current system does not support webpage screenshots. Please refer to the accessibility tree to understand the current webpage."
        else:  # adaptive decision
            if web_state.get("curr_screenshot_mode"):  # currently on
                ret["screenshot"] = _ss["boxed_screenshot"]
            else:
                ret["screenshot_note"] = "The current system's screenshot mode is off. If you need the screenshots, please use the corresponding action to turn it on."
        # --
        if suffix:
            ret = {k+suffix: v for k, v in ret.items()}
        return ret

    def _prep_html_md(self, web_state):
        _IGNORE_LINE_LEN = 7  # ignore md line if <= this
        _LOCAL_WINDOW = 2  # -W -> +W
        _budget = self.html_md_budget
        if _budget <= 0:
            return ""
        # --
        axtree, html_md = web_state["current_accessibility_tree"], web_state.get("html_md", "")
        # first locate raw texts from axtree
        axtree_texts = []
        for line in axtree.split("\n"):
            m = re.findall(r"(?:StaticText|link)\s+'(.*)'", line)
            axtree_texts.extend(m)
        # then locate to the html ones
        html_lines = [z for z in html_md.split("\n") if z.strip() and len(z) > _IGNORE_LINE_LEN]
        hit_lines = set()
        _last_hit = 0
        for one_t in axtree_texts:
            _curr = _last_hit
            while _curr < len(html_lines):
                if one_t in html_lines[_curr]: # hit
                    hit_lines.update([ii for ii in range(_curr-_LOCAL_WINDOW, _curr+_LOCAL_WINDOW+1) if ii>=0 and ii<len(html_lines)])  # add local window
                    _last_hit = _curr
                    break
                _curr += 1
        # get the contents
        _last_idx = -1
        _all_addings = []
        _all_adding_lines = []
        for line_idx in sorted(hit_lines):
            if _budget < 0:
                break
            _line = html_lines[line_idx].rstrip()
            adding = f"...\n{_line}" if (line_idx > _last_idx+1) else _line
            _all_addings.append(adding)
            _all_adding_lines.append(line_idx)
            _budget -= len(adding.encode())  # with regard to bytes!
            _last_idx = line_idx
        while _budget > 0:  # add more lines if we still have budget
            _last_idx = _last_idx + 1
            if _last_idx >= len(html_lines):
                break
            _line = html_lines[_last_idx].rstrip()
            _all_addings.append(_line)
            _all_adding_lines.append(_last_idx)
            _budget -= len(_line.encode())  # with regard to bytes!
        if _last_idx < len(html_lines):
            _all_addings.append("...")
        final_ret = "\n".join(_all_addings)
        return final_ret

    def set_multimodal(self, use_multimodal):
        if use_multimodal is not None:
            self.use_multimodal = use_multimodal

    def get_multimodal(self):
        return self.use_multimodal
