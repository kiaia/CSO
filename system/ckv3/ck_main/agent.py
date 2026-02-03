#

import time
import re
import random
from functools import partial
import multiprocessing as mp

from ..agents.agent import MultiStepAgent, register_template, AgentResult
from ..agents.tool import StopTool, AskLLMTool, SimpleSearchTool
from ..agents.utils import zwarn, GET_ENV_VAR
from ..ck_web.agent import WebAgent
try:
    from ..ck_web2.agent import SmolWebAgent  # an alternative one
except:
    SmolWebAgent = None
from ..ck_file.agent import FileAgent
from .prompts import PROMPTS as CK_PROMPTS

# --
class CKAgent(MultiStepAgent):
    def __init__(self, **kwargs):
        # note: this is a little tricky since things will get re-init again in super().__init__
        # sub-agents
        if GET_ENV_VAR("USE_SMOL_WEB", df=False):
            self.web_agent = SmolWebAgent()  # sub-agent for web
        else:
            self.web_agent = WebAgent()  # sub-agent for web
        self.file_agent = FileAgent()
        self.tool_ask_llm = AskLLMTool()
        self.tool_simple_search = SimpleSearchTool()
        feed_kwargs = dict(
            name="ck_agent",
            description="Cognitive Kernel, an initial autopilot system.",
            templates={"plan": "ck_plan", "action": "ck_action", "end": "ck_end", "aggr": "ck_aggr"},  # template names (no need of END here since we do NOT use __call__ for this)
            active_functions=["web_agent", "file_agent", "stop", "ask_llm", "simple_web_search"],  # enable the useful modules
            sub_agent_names=["web_agent", "file_agent"],  # note: another tricky point, use name rather than the objects themselves
            tools=[StopTool(agent=self), self.tool_ask_llm, self.tool_simple_search],  # add related tools
            max_steps=16,  # still give it more steps
            max_time_limit=4200,  # 70 minutes
            exec_timeout_with_call=1000,  # if calling sub-agent
            exec_timeout_wo_call=200,  # if not calling sub-agent
            max_search_results=7,  # max search results from search API
        )
        feed_kwargs.update(kwargs)
        # our new args
        self.step_mrun = 1  # step-level multiple run to do ensemble
        self.mrun_pool_size = 5  # max pool size for parallel running
        self.mrun_multimodal_count = 0  # how many runs to go with multimodal-web
        # --
        register_template(CK_PROMPTS)  # add web prompts
        super().__init__(**feed_kwargs)
        self.tool_ask_llm.set_llm(self.model)  # another tricky part, we need to assign LLM later
        self.tool_simple_search.set_llm(self.model)
        self.tool_simple_search.set_max_results(feed_kwargs["max_search_results"])
        # --

    def get_function_definition(self, short: bool):
        raise RuntimeError("Should NOT use CKAgent as a sub-agent!")

    def _super_step_action(self, _id: int, need_sleep: bool, action_res, action_input_kwargs, **kwargs):
        if need_sleep and _id:
            time.sleep(5 * int(_id))  # do not run them all at once!
        # --
        if _id is None:  # not multiple run mode
            ret = super().step_action(action_res, action_input_kwargs, **kwargs)
        else:
            _old_multimodal, _old_seed = self.web_agent.get_multimodal(), self.get_seed()
            _new_multimodal, _new_seed = ("auto" if int(_id) < self.mrun_multimodal_count else "off"), (_old_seed + int(_id))
            try:
                self.web_agent.set_multimodal(_new_multimodal)
                self.set_seed(_new_seed)
                ret = super().step_action(action_res, action_input_kwargs, **kwargs)
            finally:
                self.web_agent.set_multimodal(_old_multimodal)
                self.set_seed(_old_seed)
        # --
        return ret

    def step_action(self, action_res, action_input_kwargs, **kwargs):
        _need_multiple = any(f"{kk}(" in action_res["code"] for kk in ["web_agent", "file_agent", "ask_llm"])  # tools that might benefit from multiple running
        if self.step_mrun <= 1 or (not _need_multiple):  # just run once
            return self._super_step_action(None, False, action_res, action_input_kwargs, **kwargs)
        else:  # multiple run and aggregation
            _need_sleep = ("web_agent(" in action_res["code"])  # do not run web_agent all at once!
            with mp.Pool(min(self.mrun_pool_size, self.step_mrun)) as pool:  # note: no handle of errors here since the wraps (including the timeout) will be inside each sub-process
                # all_results = pool.map(partial(self._super_step_action, need_sleep=_need_sleep, action_res=action_res, action_input_kwargs=action_input_kwargs, **kwargs), list(range(self.step_mrun)))
                all_results = pool.map(ck_step_action, [(self, _id, _need_sleep, action_res, action_input_kwargs, kwargs) for _id in range(self.step_mrun)])
            # aggregate results
            aggr_res = None
            try:
                _aggr_inputs = action_input_kwargs.copy()
                _aggr_inputs["current_step"] = f"Thought: {action_res.get('thought')}\nAction: ```\n{action_res.get('code')}```"
                _aggr_inputs["result_list"] = "\n".join([f"### Result {ii}\n{rr}\n" for ii, rr in enumerate(all_results)])
                aggr_messages = self.templates["aggr"].format(**_aggr_inputs)
                aggr_response = self.step_call(messages=aggr_messages, session=None)  # note: for simplicity no need session info here for aggr!
                aggr_res = self._parse_output(aggr_response)
                if self.store_io:  # further storage
                    aggr_res.update({"llm_input": aggr_messages, "llm_output": aggr_response})
                _idx_str = re.findall(r"print\(.*?(\d+).*?\)", aggr_res["code"])
                _sel = int(_idx_str[-1])
                assert _sel >= 0 and _sel < len(all_results), f"Out of index error for selection index {_sel}"  # detect out of index error!
            except Exception as e:
                zwarn(f"Error when doing selection: {aggr_res} -> {e}")
                _sel = 0  # simply select the first one
            _ret = AgentResult(repr=repr(all_results[_sel]), sel_aggr=aggr_res, sel_cands=all_results, sel_idx=_sel)  # store all the information!
            return _ret
        # --

# --
# make it a top-level function
def ck_step_action(args):
    ck, _id, need_sleep, action_res, action_input_kwargs, kwargs = args
    return ck._super_step_action(_id, need_sleep, action_res, action_input_kwargs, **kwargs)
# --
