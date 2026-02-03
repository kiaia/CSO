#

# a simple wrapper for LLM calling

import time
import requests
from copy import deepcopy
import time
from .utils import wrapped_trying, rprint, GET_ENV_VAR, KwargsInitializable
import re
from openai import OpenAIError, AzureOpenAI
from ..ck_main.gaia_scorer import question_scorer
from ..agents.model import OpenaiHelper, Boto3Helper, LLM
from .evaluator_prompt import prompt_dict
from langchain_openai import AzureChatOpenAI
from langchain.evaluation import load_evaluator
import os
import json
from collections import Counter

import string
import warnings


GRADER_TEMPLATE = """You are given a question, a response, and a correct answer. Judge whether the response is correct or not based on the precise and unambiguous correct answer by completing the following two tasks:

# Task 1: REASONING

Explain why the response is correct or incorrect based on the correct answer, focusing only on if there are **meaningful differences** between the correct answer and the response. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than the correct answer, focus only on whether the answers match. Your response should strictly follow this format without any other content:
[REASONING]
<reasoning>
[END OF REASONING]

# Task 2: CORRECTNESS

Answer 'yes' if the response matches the correct answer, or is within a small margin of error for numerical problems, or if the predicted answer entails the ground truth, or with other minor differences (e.g., case sensitivity) that are not relevant to the question for other types of problems. 
Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the response is incorrect.

Your response should strictly follow this format without any other content:
[CORRECTNESS]
<yes|no>
[END OF CORRECTNESS]
"""

GRADER_USER_PROMPT="""
-----
Inputs:
* *Question:* \n```{question}\n\n```
* *Response:* \n```{response}\n\n```
* *Correct Answer:* \n```{correct_answer}\n\n```
"""

def question_scorer_llm(response, ground_truth, question ):
    llm_target = GET_ENV_VAR("EVALUATOR_LLM", df="gpt:gpt-4.1")
    llm = LLM(call_target=llm_target)
    if response is None:
        response = "None"
    if ground_truth is None:
        ground_truth = "None"
    if question is None:
        question = "None"
    if response is None:
        return 0
    max_retries = 3
    for attempt in range(max_retries):
        try:
            grading_response = llm(get_messages(GRADER_USER_PROMPT.format(question=question, response=response, correct_answer=ground_truth), 
                                                GRADER_TEMPLATE))
            match = grading_response.split("[CORRECTNESS]")[1].split("[END OF CORRECTNESS]")[0].strip()
            return "yes" in match.lower()
        except Exception as e:
            print(f"Error during evaluation: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
            else:
                print(e)
                return question_scorer(model_answer=response, ground_truth=ground_truth)


def get_prompt(prompt_name):
    """Loads the system prompt from the given JSON file."""
    return prompt_dict[prompt_name]
    


# filter out final action message and end message if they are None.
def rule_filter_final_action_message(final_message):
    if not 'stop' in final_message:
        return True
    patterns = [r'stop.*not found', r'stop.*none', r'stop.*\'\'', r'stop.*""']
    return any(bool(re.search(p, final_message, re.IGNORECASE | re.DOTALL)) for p in patterns)

def rule_filter_end_message(end_message):
    patterns = [r'output.*not found', r'output.*none', r'output.*\'\'', r'output.*""']
    return any(bool(re.search(p, end_message, re.IGNORECASE | re.DOTALL)) for p in patterns)


def remove_keys(d, keys_to_remove=["boxed_screenshot", "llm_input", "state", "llm_output", "plan", "info", "snapshot", "browser_id", "page_id", "_orig", "current_has_cookie_popup", "expanded_part", "curr_step", "curr_screenshot_mode", "total_actual_step", "num_revert_state", "answer", "file_state_before", "web_state_before"]):
    """
    Recursively removes specified keys from a nested dictionary.
    
    Parameters:
    - d (dict): The input dictionary.
    - keys_to_remove (list): List of keys to remove from the dictionary at all levels.
    
    Returns:
    - dict: A new dictionary with the specified keys removed.
    """
    if not isinstance(d, dict):
        return d  # If not a dictionary, return the object itself
    
    # Create a new dictionary without the keys to be removed
    result = {}
    for key, value in d.items():
        if key not in keys_to_remove:
            if isinstance(value, dict):
                result[key] = remove_keys(value, keys_to_remove)  # Recursively handle nested dictionaries
            elif isinstance(value, list):
                # If it's a list, process each item in the list
                result[key] = [remove_keys(item, keys_to_remove) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
    return result


def get_messages(prompt, system="You are a helpful assistant.", image_urls=None):
    """
    Constructs a message list for the OpenAI API based on the provided prompt and system message.
    
    Parameters:
    - prompt (str): The user input or question.
    - system (str): The system message to guide the assistant's behavior.
    
    Returns:
    - list: A list of messages formatted for the OpenAI API.
    """
    if "gpt" in GET_ENV_VAR("EVALUATOR_LLM", df=""):
        model = "gpt"
    else:
        model = "claude"
    if not image_urls:
        # if model == "gpt":
            return [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        # else:
        #     return [
        #         {
        #             "role": "developer",
        #             "content": [
        #                 {
        #                     'text': system,
        #                 },
        #             ]
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     'text': prompt,
        #                 },
        #             ]
        #         }
        #     ]
    else:
        return [
            {
                "role": "developer",
                "content": system
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    } for image_url in image_urls
                ]
            }
        ]
    




class Evaluator(KwargsInitializable):
    '''
        need to configure:
            AZURE_OPENAI_API_VERSION
            OPENAI_API_TYPE=azure_ai
            AZURE_OPENAI_ENDPOINT
            AZURE_INFERENCE_ENDPOINT
            AZURE_INFERENCE_CREDENTIAL
            AZURE_OPENAI_API_KEY
            AZURE_INFERENCE_CREDENTIAL
    # export EVALUATOR_LLM=gpt-4.1
    '''
    def __init__(self, **kwargs):
        # basics
        self.eval_method = ""
        # --
        super().__init__(**kwargs)  # init
        self.helper = LLM(call_target=GET_ENV_VAR("EVALUATOR_LLM"))

        # for langchain
        if GET_ENV_VAR("OPENAI_API_TYPE") == "azure_ai":
            self.llm = AzureChatOpenAI(
                azure_deployment=GET_ENV_VAR("LANGCHAIN_LLM"),
                api_version=GET_ENV_VAR("AZURE_OPENAI_API_VERSION", df="2024-02-01"),
                temperature=0.0
            )

            self.cot_qa_evaluator = load_evaluator("cot_qa", llm=self.llm)
        else:
            # raise NotImplementedError
            self.llm = None

    def summarize(self, inst):
        log_inst = remove_keys(inst, keys_to_remove=["boxed_screenshot", "llm_input", "state", "llm_output", "info", "snapshot", "browser_id", "page_id", "_orig", "current_has_cookie_popup", "expanded_part", "curr_step", "curr_screenshot_mode", "total_actual_step", "num_revert_state", "answer", "file_state_before", "task", "web_state_before", "eval"])
        # ref = inst["_orig"]["Annotator Metadata"]["Steps"]

        # summarize
        msg = "This is the trajectory of an agent completing a task, for each step the agent takes, please summarize the agent's action, key observations and information obtained after the action. If the trajectory includes detailed reasoning process, please also inlude the reasoning process in your summary. Your response should be in this format without any additional contents:\n\nStep 1: \nAction 1: <action>\nObservation 1: <observation>\nAction 2: <action>\nObservation 2: <observation>\n...\n\nStep 2: \nAction 1: <action>\nObservation 1: <observation>\nAction 2: <action>\nObservation 2: <observation>\n...\n\nStep 3: \nAction 1: <action>\nObservation 1: <observation>\nAction 2: <action>\nObservation 2: <observation>\n...\n\nHere is the trajectory:\n\n"
        # helper = OpenaiHelper() if "gpt" in GET_ENV_VAR("EVALUATOR_LLM") else Boto3Helper()
        for i in range(2):
            try:
                # log = helper.call_chat(get_messages(msg + str(log_inst)), model="gpt-4.1", max_tokens=20000, temperature=0.2, top_p=0.95, stop=None, stream=False, response_format=None)
                log = self.helper(get_messages(msg + str(log_inst)))
                break
            except Exception as e:
                print(f"Error summarizing the instance: {e}")
                log = str(log_inst)
                time.sleep(5)  # wait for a while before retrying

        return log

    def worker_detect_ask_llm(self, t):
        for step in t['steps']:
            if bool(re.search(r'ask_llm\((.*?)\)', step['action']['code'])):
                if '1' in self.determine_force_ask_llm(step['action']['thought']):
                    return True
        return False

    def determine_force_ask_llm(self, thought):
        # helper = OpenaiHelper() if "gpt" in GET_ENV_VAR("EVALUATOR_LLM") else Boto3Helper()
        while True:
            try:
                msg = get_messages("Thought: " + thought, system=get_prompt("ask_llm_system_prompt"))
                response = self.helper(msg)
                return response
            except Exception as e:
                print(e)
                try:
                    if any(e.body['innererror']['content_filter_result'][r]['filtered'] for r in e.body['innererror']['content_filter_result']):
                        return ''
                except:
                    pass
                if type(e).__name__ == 'RateLimitError':
                    time.sleep(10)
                elif type(e).__name__ == 'APIError':
                    time.sleep(15)
                elif type(e).__name__ == 'InvalidRequestError':
                    return ''
                else:
                    time.sleep(10)

    def cot_qa_evaluate(self, item):
        pred, ref, task = item['pred'], item['gold'], item['task']
        if pred is None:
            return 0
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return self.cot_qa_evaluator.evaluate_strings(
                    prediction=pred,
                    input=task,
                    reference=ref,
                )['score']
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                else:
                    print(e)
                    return 0
        
    def evaluate_with_answer(self, session, answer_gold, task, evaluation_method):
        # return True if failed (score == 0)
        try:
            answer_pred = str(session["steps"][-1]["end"]["final_results"]["output"])
        except:
            answer_pred = "error"
        if evaluation_method == "em":
            _this_corr = int(question_scorer(model_answer=answer_pred, ground_truth=answer_gold))
            return _this_corr == 0
        elif evaluation_method == "llm_score":
            _this_corr = self.cot_qa_evaluate({"pred": answer_pred, "gold": answer_gold, "task": task})
            return _this_corr == 0
        
    def gpt_judge(self, task, pred, traj):
        """
        Processes a single task by extracting the required information, sending it for verification,
        and parsing the response.

        Args:
            data (dict): The task data to be processed.

        Returns:
            dict: A dictionary containing task ID, task description, prediction, gold, verification result, and explanation.
        """
        
        prompt = f"Here is a task description: {task}\n\nHere is an unverified answer: {pred} \n\nHere is the trajectory: {traj} \n\nPlease first provide a concise explanation, then if the unverified answer is correct, return '==yes==', otherwise return '==no=='.\n\n"
        messages = get_messages(prompt, system=get_prompt("gpt_judge_heuristic_with_traj"))

        ans = "yes"
        for i in range(5):
            try:
                ans = self.helper(messages)
                break
            except Exception as e:
                print(f"Error processing task ID {id}: {e}")
                continue

        def parse_answer(ans):
            """Parses the verification response."""
            has_yes = "==yes==" in ans.strip().lower()
            has_no = "==no==" in ans.strip().lower()
            if has_yes and not has_no:
                return "yes"
            elif has_no and not has_yes:
                return "no"
            else:
                return "yes"
        
        explanation = ans.split("<think>")[1].split("</think>")[0].strip() if "<think>" in ans else "na"
        
        return parse_answer(ans), {
            "trajectory": traj,
            "pred": pred,
            "verification": parse_answer(ans),
            "explanation": explanation,
        }
    
    def detect_failure(self, session, evaluation_type):
        failed_to_answer = False
        # final action message
        action_dict = deepcopy(session['steps'][-1]['action'])
        final_message = action_dict['llm_output']
        # end message: output formatting
        end_messages = []
        for i in range(len(session['steps'])):
            if 'end' in session['steps'][i]:
                msg = deepcopy(session['steps'][i]['end']['llm_input'])
                msg.append({"role":"assistant", "content":session['steps'][i]['end']['llm_output']})
                end_messages.append(msg)
        
        if len(end_messages) == 0:
            failed_to_answer = True
        end_message = end_messages[-1][-1]['content']

        if rule_filter_final_action_message(final_message) or rule_filter_end_message(end_message):
            failed_to_answer = True
        
        if evaluation_type == "no_answer":
            return failed_to_answer, "No answer is obtained in the previous try." if failed_to_answer else None
        
        if evaluation_type == "no_answer+no_ask_llm":
            if self.worker_detect_ask_llm(session):
                return True, None
            return failed_to_answer, "Some operations in the previous try failed."
        
        if "gpt_judge" in evaluation_type:
            agent_ans = session["steps"][-1]["end"]["final_results"]["output"]
            agent_traj = self.summarize(session)
            feedback = self.gpt_judge(session["task"], agent_ans, agent_traj)
            if feedback[0] == "no":
                return True, feedback[1]
            return False, None
        else:
            return False, None
        
    def extract_answer_and_log(self, session):
        """Extracts the answer and log from a given instance."""
        ans = session["steps"][-1]["end"]["final_results"]["output"]
        log = self.summarize(session)
        return ans, log
        
    def construct_prompt(self, session_list):
        """Constructs the prompt based on the task and instance list. Optionally reduces log length."""
        task = session_list[0]["task"]
        prompt = "===== Begin of task =====\n\n"
        prompt += task
        for i in range(len(session_list)):
            prompt += f"===== Begin of solution {i} =====\n\n"
            ans, log = self.extract_answer_and_log(session_list[i])
            prompt += f"Answer: {ans}\n\n"
            prompt += f"Log:\n {log}\n\n"
        return prompt
        
    def ensemble(self, session_list):
        prompt = self.construct_prompt(session_list)
        candidates = list(enumerate([x["steps"][-1]["end"]["final_results"]["output"] for x in session_list]))
        try:
            ans = self.helper(get_messages(prompt, system=get_prompt("gpt_chooser"))) 
            print(f"Ensemble answer: {ans}")
        except:
            strings = [item[1] for item in candidates]
            count = Counter(strings)
            # Find the majority string
            majority_string = count.most_common(1)[0][0]
            # Find the index of the majority string
            majority_index = next(i for i, s in candidates if s == majority_string)
            ans = f"<think>Majority voting</think><choice>{majority_index}</choice>"

        if not "<choice>" in ans:
            ans += "<choice>"
        if not "<think>" in ans:
            ans += "<think>"

        return int(ans.split("<choice>")[1].split("</choice>")[0].strip())

        # return {
        #     "choice": int(ans.split("<choice>")[1].split("</choice>")[0].strip()),
        #     "explanation": ans.split("<think>")[1].split("</think>")[0].strip(),
        #     "log": prompt,
        # }
if __name__ == "__main__":
    # for testing
    evaluator = Evaluator()
    task = "What is the capital of France?"
    pred = "Paris"
    traj = "Step 1: Action 1: Search for the capital of France. Observation 1: The capital of France is Paris."
    result = evaluator.gpt_judge(task, pred, traj)
    print(result)

        
