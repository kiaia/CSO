#

import json
from ..agents.agent import MultiStepAgent, register_template, ActionResult
from ..agents.utils import zwarn, GET_ENV_VAR, have_images_in_messages
from ..agents.model import LLM

from .utils import FileEnv
from .prompts import PROMPTS as FILE_PROMPTS

class FileAgent(MultiStepAgent):
    def __init__(self, **kwargs):
        # note: this is a little tricky since things will get re-init again in super().__init__
        feed_kwargs = dict(
            name="file_agent",
            description="A file agent helping to parse and process (a) file(s) to solve a specific task.",
            templates={"plan": "file_plan", "action": "file_action", "end": "file_end"},  # template names
            max_steps=16,
        )
        feed_kwargs.update(kwargs)
        self.file_env_kwargs = {}  # kwargs for file env
        self.check_nodiff_steps = 3  # if for 3 steps, we have the same file page, then explicitly indicating this!
        self.max_file_read_tokens = int(GET_ENV_VAR("MAX_FILE_READ_TOKENS", df=3000))
        self.max_file_screenshots = int(GET_ENV_VAR("MAX_FILE_SCREENSHOT", df=2))
        self.file_env_kwargs['max_file_read_tokens'] = self.max_file_read_tokens
        self.file_env_kwargs['max_file_screenshots'] = self.max_file_screenshots
        self.model_multimodal = LLM(_default_init=True)  # multimodal model
        # --
        register_template(FILE_PROMPTS)  # add web prompts
        super().__init__(**feed_kwargs)
        self.file_envs = {}  # session_id -> ENV
        self.current_session = None
        self.ACTIVE_FUNCTIONS.update(stop=self._my_stop, load_file=self._my_load_file, read_text=self._my_read_text, read_screenshot=self._my_read_screenshot, search=self._my_search)
        # --

    # note: a specific stop function!
    def _my_search(self, file_path: str, key_word_list: list):
        return ActionResult(f"search({file_path}, {key_word_list})")

    def _my_stop(self, answer: str = None, summary: str = None, output: str = None):
        if output:
            ret = f"Final answer: [{output}] ({summary})"
        else:
            ret = f"Final answer: [{answer}] ({summary})"
        self.put_final_result(ret)  # mark end and put final result
        return ActionResult("stop", ret)

    def _my_load_file(self, file_path: str):
        return ActionResult(f'load_file({file_path})')

    def _my_read_text(self, file_path: str, page_id_list: list):
        return ActionResult(f"read_text({file_path}, {page_id_list})")

    def _my_read_screenshot(self, file_path: str, page_id_list: list):
        return ActionResult(f"read_screenshot({file_path}, {page_id_list})")
    
    def get_function_definition(self, short: bool):
        if short:
            return "- def file_agent(task: str, file_path_dict: dict = None) -> Dict:  # Processes and analyzes one or more files to accomplish a specified task, with support for various file types such as PDF, Excel, and images."
        else:
            return """- file_agent
```python
def file_agent(task: str, file_path_dict: dict = None) -> dict:
    \""" Processes and analyzes one or more files to accomplish a specified task.
    Args:
        task (str): A clear description of the task to be completed. If the task requires a specific output format, specify it here.
        file_path_dict (dict, optional): A dictionary mapping file paths to short descriptions of each file.
            Example: {"./data/report.pdf": "Annual financial report for 2023."}
            If not provided, file information may be inferred from the task description.
    Returns:
        dict: A dictionary with the following structure:
            {
                'output': <str>  # The well-formatted answer to the task.
                'log': <str>     # Additional notes, processing details, or error messages.
            }
    Notes:
        - If the task specifies an output format, ensure the `output` field matches that format.
        - Supports a variety of file types, including but not limited to PDF, Excel, images, etc.
        - If no files are provided or if files need to be downloaded from the Internet, return control to the external planner to invoke a web agent first.
    Example:
        >>> answer = file_agent(task="Based on the files, what was the increase in total revenue from 2022 to 2023?? (Format your output as 'increase_percentage'.)", file_path_dict={"./downloadedFiles/revenue.pdf": "The financial report of the company XX."})
        >>> print(answer)  # directly print the full result dictionary
    \"""
```"""

    def __call__(self, task: str, file_path_dict: dict = None, **kwargs):  # allow *args styled calling
        return super().__call__(task, file_path_dict=file_path_dict, **kwargs)

    def init_run(self, session):
        super().init_run(session)
        _id = session.id
        assert _id not in self.file_envs
        _kwargs = self.file_env_kwargs.copy()
        if session.info.get("file_path_dict"):
            _kwargs["starting_file_path_dict"] = session.info["file_path_dict"]
        self.file_envs[_id] = FileEnv(**_kwargs)
        self.current_session = session

    def end_run(self, session):
        ret = super().end_run(session)
        _id = session.id
        self.file_envs[_id].stop()
        del self.file_envs[_id]  # remove web env
        return ret

    def step_prepare(self, session, state):
        self.current_session = session
        _input_kwargs, _extra_kwargs = super().step_prepare(session, state)
        _file_env = self.file_envs[session.id]

        _input_kwargs["max_file_read_tokens"] = _file_env.max_file_read_tokens
        _input_kwargs["max_file_screenshots"] = _file_env.max_file_screenshots
        page_result = self._prep_page(_file_env.get_state()) # current file content
        _input_kwargs["textual_content"] = page_result['textual_content']
        _input_kwargs["file_meta_data"] = page_result['file_meta_data']
        _input_kwargs["loaded_files"] = page_result['loaded_files']
        _input_kwargs["visual_content"] = page_result['visual_content']
        _input_kwargs["image_suffix"] = page_result['image_suffix']
        if not page_result["error_message"] is None:
            _input_kwargs["textual_content"] += "Note the error message:" + page_result['error_message']


        if session.num_of_steps() > 1:  # has previous step
            _prev_step = session.get_specific_step(-2)  # the step before
            _input_kwargs["textual_content_old"] = self._prep_page(_prev_step["action"]["file_state_before"])["textual_content"]  # old web page
        else:
            _input_kwargs["textual_content_old"] = "N/A"
        _extra_kwargs["file_env"] = _file_env

        return _input_kwargs, _extra_kwargs

    def step_action(self, action_res, action_input_kwargs, file_env=None, **kwargs):
        action_res["file_state_before"] = file_env.get_state()  # inplace storage of the web-state before the action
        _rr = super().step_action(action_res, action_input_kwargs)  # get action from code execution
        if isinstance(_rr, ActionResult):
            action_str, action_result = _rr.action, _rr.result
        else:
            action_str = self.get_obs_str(None, obs=_rr, add_seq_enum=False)
            action_str, action_result = "nop", action_str.strip()  # no-operation
        # --
        try:  # execute the action on the browser
            step_result = file_env.step_state(action_str)
            ret = action_result if action_result is not None else step_result  # use action result if there are direct ones
            # return f"File agent step: {action_str.strip()}"
        except Exception as e:
            zwarn("file_env execution error!" + f"\nFile agent error: {e} for {_rr}")
            ret = f"File agent error: {e} for {_rr}"
        return ret

    def step_call(self, messages, session, model=None):
        _use_multimodal = session.info.get("use_multimodal", False) or have_images_in_messages(messages)
        if model is None:
            model = self.model_multimodal if _use_multimodal else self.model  # use which model?
        response = model(messages)
        return response

    # --
    # other helpers

    def _prep_page(self, file_state):
        _ss = file_state

        _ret = {"loaded_files": _ss["loaded_files"],
                "file_meta_data":_ss["file_meta_data"],
                "textual_content":_ss["textual_content"],
                "visual_content":None,
                "image_suffix":None,
                "error_message":None}


        if _ss["error_message"]:
            # _ret = _ret + "\n(Note: " + _ss["error_message"] + ")"
            _ret["error_message"] = _ss["error_message"]
        if _ss["visual_content"]:
            _ret["visual_content"] = _ss["visual_content"]
            _ret["image_suffix"] = _ss["image_suffix"]
        
        return _ret
