import json
import argparse
from itertools import chain
from copy import deepcopy
import os
from tqdm import tqdm
import os
from openai import AzureOpenAI

import os
import time
import re
import json
from openai import OpenAIError, AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import AzureChatOpenAI
from langchain.evaluation import load_evaluator


def execute_with_print_ask_llm(code_string):
    captured_arg = None
    
    # execute ask_llm to print the argument
    # such that we can directly get the argument.
    def ask_llm(argument):
        nonlocal captured_arg
        captured_arg = argument
        return argument
    
    try:
        local_vars = {'ask_llm': ask_llm, 'print': print}
        
        exec(code_string, {'__builtins__': {}}, local_vars)
        
        return captured_arg
        
    except Exception as e:
        print(f"error: {e}")
        return None

def get_ask_llm(result):
    all_ask_llm_arguments = []
    all_ask_llm_observations = []
        
    for step in result['session']['steps']:
        if 'ask_llm' in step['action']['code']:
            argument = execute_with_print_ask_llm(step['action']['code'])
            if argument:
                observation = step['action']['observation']
                if isinstance(observation, str):
                    observation = observation.strip()
                elif isinstance(observation, list) and all(isinstance(x, str) for x in observation):
                    observation = " ".join(observation).strip()
                else:
                    observation = ""
                if not observation == "":
                    all_ask_llm_arguments.append(argument)
                    all_ask_llm_observations.append(observation)

    # prepare messages

    ask_llm_messages = []
    for argument, observation in zip(all_ask_llm_arguments, all_ask_llm_observations):
        message = [{"role": "system", "content":"You are a helpful assistant. Answer the user's query with your internal knowledge. Ensure to follow the required output format if specified."}]
        message.append({"role": "user", "content": argument})
        message.append({"role": "assistant", "content": observation})
        ask_llm_messages.append(message)
    return ask_llm_messages

class LangChainEvaluator:
    def __init__(self):
        # llm = AzureChatOpenAI(
        #     azure_deployment="gpt-4.1",  # or your deployment
        #     api_version="2025-01-01-preview",  # or your api version
        #     temperature=0)

        llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1",  # or your deployment
            api_version="2025-01-01-preview",  # or your api version
            temperature=0)

        self.evaluator = load_evaluator("cot_qa",llm=llm)

    def evaluate_item(self, item, max_retries=10):
        if item['eval']['gold'] is None:
            return None
        for attempt in range(max_retries):
            try:
                return self.evaluator.evaluate_strings(
                    prediction=item['eval']['pred'],
                    input=item['task'],
                    reference=item['eval']['gold'],
                )
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                else:
                    print(e)
                    return None


def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data

def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def get_text_sft_data(item):
    action_messages = []
    
    for i in range(len(item['session']['steps'])):
        action_dict = deepcopy(item['session']['steps'][i]['action'])
        msg = action_dict['llm_input']
        msg.append({"role":"assistant","content":action_dict['llm_output']})
        action_messages.append(msg)
    
    end_messages = []
    for i in range(len(item['session']['steps'])):
        if 'end' in item['session']['steps'][i]:
            msg = deepcopy(item['session']['steps'][i]['end']['llm_input'])
            msg.append({"role":"assistant", "content":item['session']['steps'][i]['end']['llm_output']})
            end_messages.append(msg)
        

    planning_messages = []
    for i in range(len(item['session']['steps'])):
        plan_dict = deepcopy(item['session']['steps'][i]['plan'])
        msg = plan_dict['llm_input']
        #print(len(msg), msg)
        msg.append({"role":"assistant","content":plan_dict['llm_output']})
        #print(len(msg), msg)
        #print(msg[0]["content"],msg[0])
        planning_messages.append(msg)


    sub_action_messages = []
    sub_end_messages = []
    for i in range(len(item['session']['steps'])):
        ori_action_dict = deepcopy(item['session']['steps'][i]['action'])
        if "observation" in ori_action_dict and ori_action_dict["observation"]!=[]:
            ob=ori_action_dict["observation"]
            if type(ob)==dict and "session" in ob and 'steps' in ob['session']:
                #print(ob,"\n\n")
                subagent_messages = [] 
                subagent_end_messages = []
                for j in range(len(ob['session']['steps'])):
                    action_dict = deepcopy(ob['session']['steps'][j]['action'])
                    msg = action_dict['llm_input']
                    #filter data contain image url
                    temp=0
                    for p in msg:
                        if type(p["content"])!=str:
                            temp=1
                    if temp==1:
                        continue
                    msg.append({"role":"assistant","content":action_dict['llm_output']})
                    #print(len(msg), msg)
                    #print(msg[0]["content"],msg[0])
                    
                    subagent_messages.append(msg)

                for j in range(len(ob['session']['steps'])):
                    if 'end' in ob['session']['steps'][j]:
                        msg = deepcopy(ob['session']['steps'][j]['end']['llm_input'])
                        msg.append({"role":"assistant","content":ob['session']['steps'][j]['end']['llm_output']})
                        subagent_end_messages.append(msg)
        
                sub_action_messages.append(subagent_messages)
                sub_end_messages.append(subagent_end_messages)
    sub_planning_messages = []
    for i in range(len(item['session']['steps'])):
        ori_action_dict = deepcopy(item['session']['steps'][i]['action'])
        if "observation" in ori_action_dict and ori_action_dict["observation"]!=[]:
            ob=ori_action_dict["observation"]
            if type(ob)==dict and "session" in ob and 'steps' in ob['session']:
                subagent_messages = [] 
                for j in range(len(ob['session']['steps'])):
                    plan_dict = deepcopy(ob['session']['steps'][j]['plan'])
                    msg = plan_dict['llm_input']
                    #filter data contain image url
                    temp=0
                    for p in msg:
                        if type(p["content"])!=str:
                            temp=1
                    if temp==1:
                        continue
                    msg.append({"role":"assistant","content":plan_dict['llm_output']})
                    #print(len(msg), msg)
                    #print(msg[0]["content"],msg[0])
                    subagent_messages.append(msg)
                sub_planning_messages.append(subagent_messages)
                
    return action_messages, planning_messages, end_messages, sub_action_messages, sub_planning_messages, sub_end_messages

def get_multimodal_sft_data(item, screenshot_path):
    action_messages = []
    planning_messages = []
    for i in range(len(item['session']['steps'])):
        
        action_dict = deepcopy(item['session']['steps'][i]['action'])
        browser_id = action_dict['web_state_before']['browser_id']
        msg = action_dict['llm_input']
        
        file_exists = False
        try:
            # move the screenshot to destination
            os.system(f"cp {screenshot_path}/{browser_id}@@0@@{i}.png {screenshot_destination_path}/{browser_id}@@0@@{i}.png")
            os.system(f"cp {screenshot_path}/{browser_id}@@0@@{i}_with_box.png {screenshot_destination_path}/{browser_id}@@0@@{i}_with_box.png")
            file_exists = True
        except FileNotFoundError:
            print(f"File not found: {screenshot_path}/{browser_id}@@0@@{i}.png")
        except Exception as e:
            print(f"Error copying file: {e}")

        
        if file_exists:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }, 
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}.png"}},
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}_with_box.png"}},
            ]
        else:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }]


        msg.append({"role":"assistant","content":action_dict['llm_output']})
        action_messages.append(msg)


        ### planning msg

        plan_dict = deepcopy(item['session']['steps'][i]['plan'])
        msg = plan_dict['llm_input']
        if file_exists:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }, 
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}.png"}},
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}_with_box.png"}},
            ]
        else:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }]
        msg.append({"role":"assistant","content":plan_dict['llm_output']})
        planning_messages.append(msg)
        
    return action_messages, planning_messages

def get_multimodal_sft_data(item, screenshot_path):
    action_messages = []
    planning_messages = []
    for i in range(len(item['session']['steps'])):
        
        action_dict = deepcopy(item['session']['steps'][i]['action'])
        browser_id = action_dict['web_state_before']['browser_id']
        msg = action_dict['llm_input']
        
        file_exists = False
        try:
            # move the screenshot to destination
            os.system(f"cp {screenshot_path}/{browser_id}@@0@@{i}.png {screenshot_destination_path}/{browser_id}@@0@@{i}.png")
            os.system(f"cp {screenshot_path}/{browser_id}@@0@@{i}_with_box.png {screenshot_destination_path}/{browser_id}@@0@@{i}_with_box.png")
            file_exists = True
        except FileNotFoundError:
            print(f"File not found: {screenshot_path}/{browser_id}@@0@@{i}.png")
        except Exception as e:
            print(f"Error copying file: {e}")

        
        if file_exists:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }, 
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}.png"}},
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}_with_box.png"}},
            ]
        else:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }]


        msg.append({"role":"assistant","content":action_dict['llm_output']})
        action_messages.append(msg)


        ### planning msg

        plan_dict = deepcopy(item['session']['steps'][i]['plan'])
        msg = plan_dict['llm_input']
        if file_exists:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }, 
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}.png"}},
                                {"type": "image_url", "image_url": {"url": f"screenshot/{browser_id}@@0@@{i}_with_box.png"}},
            ]
        else:
            msg[-1]['content'] = [{"type": "text", "content": msg[-1]['content'] }]
        msg.append({"role":"assistant","content":plan_dict['llm_output']})
        planning_messages.append(msg)


        
    return action_messages, planning_messages

import re
# filter out final action message and end message if they are None.
def rule_filter_final_action_message(final_message):
    if not 'stop' in final_message:
        return True
    # patterns = [r'stop.*not found', r'stop.*none', r'stop.*\'\'', r'stop.*""']
    patterns = [r'stop.*not found', r'stop.*none']
    return any(bool(re.search(p, final_message, re.IGNORECASE | re.DOTALL)) for p in patterns)

def rule_filter_end_message(end_message):
    # patterns = [r'output.*not found', r'output.*none', r'output.*\'\'', r'output.*""']
    patterns = [r'output.*not found', r'output.*none']
    return any(bool(re.search(p, end_message, re.IGNORECASE | re.DOTALL)) for p in patterns)

def get_main_agent_action_counter(action_messages):

    action_cnter = {
        "web_agent": 0,
        "simple_web_search": 0,
        "file_agent": 0,
        "ask_llm": 0,
    }

    action_cnter = {k: sum([k in item[-1]['content'] if isinstance(item, list) and 'content' in item[-1] and item[-1]['content'] is not None else 0 for item in action_messages]) for k in action_cnter}
    action_cnter['other'] = len(action_messages) - sum(action_cnter.values())
    return action_cnter


system_prompt = """
You will be presented with a chain-of-thought indicating the reason of calling ask_llm function.

If the reason is because the previous failures, or the previuos web/file agent were not able to retrieve useful information, return 1, otherwise return 0.

For example, 

Thought: Previous searches for the number of new recipes in the "Innovative French Cuisine" section of the 2018 "Culinary Arts Review" have failed, even after progressively broadening the queries. Since no direct data is available, I will now use ask_llm to estimate or infer the likely number of new recipes in that section, as this is the only remaining viable approach.
All attempts to locate the official IEA Germany 2021 review or any reputable summary via web search have failed, even with the broadest queries. As suggested in the progress state, the next best step is to use ask_llm to estimate or summarize the answer based on general knowledge, clearly noting the lack of direct source and focusing on the integer-rounded percentage as requested.
Previous web searches failed to return any results, indicating that the web search tool is currently unable to retrieve the required information. Therefore, I will use ask_llm to directly obtain the highest number of islands mentioned on the Wikipedia page for the Philippines.

Output: 1

Thought: No previous steps have been taken yet. The task is to extract the sentence by reading all letters in the provided 6x6 block from left to right, row by row. I will use ask_llm to process the block and output the sentence.

Output: 0
"""


def determine_force_ask_llm(thought):
    while True:
        try:
            response = client.chat.completions.create(
                    model="gpt-4.1", # model = "deployment_name".
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Thought: " + thought},
                    ]
                )
            return response.to_dict()['choices'][0]["message"]['content']
        except Exception as e:
            print(e)
            try:
                # if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
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
    

# re-generate the response for queries in all_ask_llm_arguments
def sample_output(query):
    while True:
        try:
            response = client.chat.completions.create(
                    model="gpt-4.1", # model = "deployment_name".
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the user's query with your internal knowledge. Ensure to follow the required output format if specified."}, 
                        {"role": "user", "content": query}
                    ]
                )
            return response.to_dict()['choices'][0]["message"]['content']
        except Exception as e:
            print(e)
            try:
                # if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
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
# python convert_sft.py --input_file ../output/xxx --output_file ../output/sft/xxx --save_ask_llm --rejection_sampling_type trajectory --simple_ratio_filter 0.8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', type=str, nargs='+', required=True, help='Input JSONL file path(s)')
    parser.add_argument('--output_file', type=str, required=True, help='Output SFT JSONL file path')

    parser.add_argument('--remove_ask_llm', action='store_true', help='Whether to remove the stopped trajectories where ask_llm is performed to acquire the final answer due to previous failures.')
    parser.add_argument('--save_ask_llm', action='store_true', help='Whether to save the ask_llm results.')
    parser.add_argument('--rejection_sampling_type', choices=['none', 'llm_judge', 'em', 'trajectory'], default='none')

    parser.add_argument('--main_step_filter', type=int, default=0, help='filter out trajectories with main step < main_step_filter')
    parser.add_argument('--web_step_filter', type=int, default=0, help='filter out trajectories with web step < web_step_filter')
    parser.add_argument('--simple_ratio_filter', type=float, default=0, help='filter out trajectories where simple_web_search + ask_llm ratio > simple_ratio_filter')



    parser.add_argument('--include_screenshot', action='store_true', help='Include screenshots in the data')
    parser.add_argument('--input_screenshot_path', type=str, required=False, help='Path to screenshot saved by the web agent')
    parser.add_argument('--screenshot_destination_path', type=str, required=False, help='Destination path for screenshots')
    

    args = parser.parse_args()

    num_filter_out_by_main_step = 0
    num_filter_out_by_simple_ratio = 0
    num_filter_out_by_web_step = 0

    if args.include_screenshot:
        screenshot_destination_path = args.screenshot_destination_path
        os.makedirs(screenshot_destination_path, exist_ok=True)

    trajectories = []
    for input_file in args.input_file:
        trajectories += read_jsonl(input_file)

    if args.rejection_sampling_type == 'trajectory':
        # use what's in the trajectory
        accuracy = sum([item['eval']['corr'] == 1 for item in trajectories]) / len(trajectories)
        print("exact match accuracy:", format(accuracy, '.4g'))
        trajectories = [item for item in trajectories if item['eval']['corr'] == 1]
    elif args.rejection_sampling_type == 'em':
        raise NotImplementedError("if GAIA scorer is needed, implement it later")
    elif args.rejection_sampling_type == 'llm_judge':
        # langchain judge
        evaluator = LangChainEvaluator()
        scores = []
        with ThreadPoolExecutor(10) as executor:
            for res in tqdm(executor.map(evaluator.evaluate_item, trajectories), total=len(trajectories)):
                scores.append(res)
        trajectories = [item for item, score in zip(trajectories, scores) if score['score'] == 1]
        print("number of none in evaluation:", sum([item['score'] is None for item in scores]))
        accuracy = sum([item['score'] if item['score'] is not None else 0 for item in scores]) / len(scores)
        print("llm evaluator accuracy", format(accuracy, '.4g'))

    # client = AzureOpenAI(
    #     azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'], 
    #     api_key=os.environ['AZURE_OPENAI_API_KEY'],  
    #     api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    # )

    if args.remove_ask_llm:

        def worker_detect_ask_llm(t):
            for step in t['session']['steps']:
                if bool(re.search(r'ask_llm\((.*?)\)', step['action']['code'])):
                    if '1' in determine_force_ask_llm(step['action']['thought']):
                        return True
            return False

        filter_label = []
        with ThreadPoolExecutor(5) as executor:
            for res in tqdm(executor.map(worker_detect_ask_llm, trajectories), total=len(trajectories)):
                filter_label.append(res)
        print(f"Filtering trajectories... number of original trajectories: {len(trajectories)}, number after filter {len(trajectories)-sum(filter_label)}",)
        trajectories = [t for t, label in zip(trajectories, filter_label) if not label]

    #print(len(trajectories))
    if not args.include_screenshot:
        sft_data = []
        ask_llm_sft = []
        all_ask_llm_arguments = []
        all_ask_llm_observations = []
    
        success_traj_cnt = 0
        for i in tqdm(range(len(trajectories))):

            action_messages, \
            planning_messages,\
            end_messages, \
            sub_action_messages, \
            sub_planning_messages,\
            sub_end_messages = get_text_sft_data(trajectories[i])

            ### difficulty filter:
            if len(action_messages) < args.main_step_filter:
                num_filter_out_by_main_step += 1
                continue

            action_counter = get_main_agent_action_counter(action_messages)
            if (action_counter['simple_web_search'] + action_counter['ask_llm']) / len(action_messages) > args.simple_ratio_filter:
                num_filter_out_by_simple_ratio += 1
                continue
            
            # check sub agent types
            sub_agent_type = []
            for item in action_messages:
                flag = False
                for type_ in ["web_agent", "file_agent", "simple_web_search",  "ask_llm"]:
                    if type_ in item[-1]['content']:
                        flag = True
                        sub_agent_type.append(type_)
                        break
                if not flag:
                    sub_agent_type.append("other")
            sub_agent_step_cnt = [len(sub_action_messages[i]) for i in range(len(sub_action_messages))]

            filtered_subagent_idx = []
            for sub_i, (type_, step_cnt_) in enumerate(zip(sub_agent_type, sub_agent_step_cnt)):
                if type_ == "web_agent" and step_cnt_ < args.web_step_filter:
                    filtered_subagent_idx.append(sub_i)
                    num_filter_out_by_web_step += 1



            stopped_messages = []
            # if this task has not finished
            final_message = action_messages[-1][-1]['content']
            if len(end_messages) == 0:
                continue
            end_message = end_messages[-1][-1]['content']
            if rule_filter_final_action_message(final_message) or rule_filter_end_message(end_message):
                # failed
                continue
            else:
                stopped_messages = action_messages + planning_messages + end_messages
                success_traj_cnt += 1
            


            for subagent_i in range(len(sub_action_messages)):

                if subagent_i in filtered_subagent_idx:
                    continue

                final_message = sub_action_messages[subagent_i][-1][-1]['content']
                # check if this subagent has finished
                if len(sub_end_messages[subagent_i]) > 0:
                    end_message = sub_end_messages[subagent_i][-1][-1]['content']
                else:
                    continue
                #print(end_message)
                if not final_message or not end_message:
                    # no ending message
                    continue
                elif rule_filter_final_action_message(final_message) or rule_filter_end_message(end_message):
                    # failed to stop
                    continue
                else: 
                    stopped_messages = stopped_messages + sub_action_messages[subagent_i] + sub_planning_messages[subagent_i] + sub_end_messages[subagent_i]

            sft_data.extend(stopped_messages)

            if args.remove_ask_llm or args.save_ask_llm:
                # acquire the ask_llm keys and values as they are not saved.

                ask_llm_message = get_ask_llm(trajectories[i])
                ask_llm_sft.extend(ask_llm_message)

    else:
        raise NotImplementedError
    print("number of sft_data", len(sft_data), "number of ask_llm data", len(ask_llm_sft))
    print(f"number of filtered out traj, num_filter_out_by_main_step: {num_filter_out_by_main_step} num_filter_out_by_simple_ratio: {num_filter_out_by_simple_ratio} num_filter_out_by_web_step: {num_filter_out_by_web_step}")
    save_jsonl(sft_data, args.output_file)
    save_jsonl(ask_llm_sft, args.output_file.replace(".jsonl", ".ask_llm.jsonl"))