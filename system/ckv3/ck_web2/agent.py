#

import time
from typing import List, Dict, Optional

from smolagents import CodeAgent, ApiModel, ChatMessage, PlanningStep, ActionStep, FinalAnswerStep

from .helium_browser import *
from ..ck_web.agent import WebAgent
from ..agents.tool import AskLLMTool

AUTHORIZED_IMPORTS = ["requests", "zipfile", "os", "pandas", "numpy", "sympy", "json", "bs4", "pubchempy", "xml", "yahoo_finance", "Bio", "sklearn", "scipy", "pydub", "io", "PIL", "chess", "PyPDF2", "pptx", "torch", "datetime", "fractions", "csv", "pandas", "helium"]

class MyAPIModelAdapter(ApiModel):
    def __init__(self, llm):
        super().__init__(model_id="")
        self.llm = llm

    def create_client(self):
        return None  # return a fake one!

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        response = self.llm(**completion_kwargs)
        # response = self.client.chat.completions.create(**completion_kwargs)
        # self.last_input_token_count = response.usage.prompt_tokens
        # self.last_output_token_count = response.usage.completion_tokens
        first_message = ChatMessage.from_dict({"role": "assistant", "content": response})
        return self.postprocess_message(first_message, tools_to_call_from)

class SmolWebAgent(WebAgent):
    def __init__(self, **kwargs):
        feed_kwargs = kwargs.copy()
        super().__init__(**feed_kwargs)
        # --
        self.smolagent = self._init_smolagent()
        # --

    def _init_smolagent(self):
        setup_ask_llm_tool(AskLLMTool(llm=self.model))
        agent = CodeAgent(
            tools=[DownloadTool(), MixedSearchTool(), go_back, close_popups, search_item_ctrl_f, visit_webpage, scroll_down_window, scroll_up_window, perform_click, perform_input],
            model=MyAPIModelAdapter(self.model),  # adapter
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            step_callbacks=[save_screenshot],
            max_steps=self.max_steps,
            verbosity_level=2,
            planning_interval=4,
        )
        return agent

    # directly override this full one!
    def yield_session_run(self, session, max_steps):
        driver = initialize_driver()  # init
        task = session.task
        step_idx = 0
        for step in self.smolagent.run(task, stream=True, max_steps=max_steps):
            step_dict = step.dict()
            _step_info = {"step_idx": step_idx} | step_dict
            step_idx += 1
            if isinstance(step, FinalAnswerStep):  # put final answer
                _step_info["end"] = {"final_results": {"output": step.final_answer, "log": ""}}
            session.add_step(_step_info)
            yield _step_info
            # breakpoint()
        driver.quit()  # end
