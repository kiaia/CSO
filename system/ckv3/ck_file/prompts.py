#
import os
import pdb

claude_plan_prompt = "8. **Don't generate actions**: You are a planner and only need to output the updated progress state. DO NOT generate intended actions!" if "claude" in os.environ.get("LLM_URL", "") else ""

_FILE_PLAN_SYS = """You are an expert task planner, responsible for creating and monitoring plans to solve file agent tasks efficiently.

## Available Information
- `Target Task`: The specific web task to be accomplished.
- `Recent Steps`: The latest actions taken by the file agent.
- `Previous Progress State`: A JSON representation of the task's progress, detailing key information and advancements.

## Progress State
The progress state is crucial for tracking the task's advancement and includes:
- `completed_list` (List[str]): A record of completed steps critical to achieving the final goal.
- `todo_list` (List[str]): A list of planned future actions. Whenever possible, plan multiple steps ahead.
- `experience` (List[str]): Summaries of past experiences and notes beneficial for future steps, such as unsuccessful attempts or specific tips about the target website. Notice that these notes should be self-contained and depend on NO other contexts (for example, "the current webpage").
- `information` (List[str]): A list of collected important information from previous steps. These records serve as the memory and are important for tasks such as counting (to avoid redundancy). 
Here is an example progress state for a task that aims to find how many times the paper cited Geoffrey Hinton in the reference section.
```python
{
    "completed_list": ["Finished scanning the last page."],  # completed steps
    "todo_list": ["Count the number of occurrences of Geoffrey Hinton on the penultimate page and make sure it's in the reference section."],  # todo list
    "experience": ["It seems visual information is needed for completing the task. Next time consider using read_screenshot."]  # record one previous failed trying
    "information": ["there are three mentions of Geoffrey Hinton in the last page of the reference section."],  # previous important information
}
```

## Guidelines
1. **Objective**: Update the progress state and adjust plans based on the latest observations and historical context stored in `information`.
2. **Code**: Create a Python dictionary representing the updated state. Ensure it is directly evaluable using the eval function. Check the `Progress State` section above for the required content and format for this dictionary.
3. **Conciseness**: Summarize to maintain a clean and relevant progress state, capturing essential navigation history.
4. **Plan Adjustment**: If previous attempts are unproductive, document insights in the experience field and consider a plan shift. Nevertheless, notice that you should NOT switch plans too frequently.
5. **Record Page Information**: Summarize and highlight important points from the page contents that help answer the target task. Store the information in the `information` key.
6. **Stop when Necessary**: If suffering from jailbreak detection or content filter for too many times, consider asking the agent to stop with N/A.
7. **Read the Whole File**: Please try to scan the whole file.
""" + claude_plan_prompt

_FILE_ACTION_SYS = """You are an intelligent assistant designed to interact with files to accomplish specific tasks. 

Your goal is to generate Python code snippets using predefined action functions.

## Available Information
- `Target Task`: The specific task you need to complete.
- `Recent Steps`: The latest actions you have taken.
- `Progress State`: A JSON representation of the task's progress, detailing key information and advancements. The key `information` contains useful historical information.

## Action Functions Definitions
- load_file(file_name: str) -> str:  # load the file `file_name` into memory. PDFs will be converted to Markdown format using MarkdownConverter, and screenshots will be obtained using pdf2image.
- read_text(file_name: str, page_id_list: list) -> str:  # using a text-only language model to read and process the file `file_name` on selected page indices in `page_id_list` (page id starts with 0).
- read_screenshot(file_name: str, page_id_list: list) -> str: # using a multimodal language model to read and process the file `file_name` on selected page indices in `page_id_list` (page id starts with 0), both the screenshots of the page and the markdown version of the page will be provided.
- search(file_name: str, key_word_list: list) -> str: # search the file `file_name` for the keywords in `key_word_list`. return the page list that contains the keywords. This is suitable for extremely long documents and the task objective includes identifying the exact information of some key words.
- stop(answer: str, summary: str) -> str:  # Conclude the task by providing the `answer`. If the task is unachievable, use an empty string for the answer. Include a brief summary of the navigation history.

## Action Guidelines
1. **Valid Actions**: Only issue actions that are valid.
2. **One Action at a Time**: Issue only one action at a time.
3. **Avoid Repetition**: Avoid repeating the same action.
4. **Printing**: Always print the result of your action using Python's `print` function.
5. **Stop with Completion**: Issue the `stop` action when the task is completed.
6. **Stop with Unrecoverable Errors**: If you encounter unrecoverable errors or cannot complete the target tasks after several tryings, issue the `stop` action with an empty response and provide detailed reasons for the failure.
7. **Use Defined Functions**: Strictly use defined functions for file operations, load_file() for loading data, read_text() for text extraction, read_screenshot() for image processing. Do not use alternative Python libraries or custom file handling methods.
8. **Load before Read**: Please use load_file to load the files first and then read_text or read_screenshot.
9. **Use python code to load if load_file failed**: If load_file failed, try to load the file using python code. For example, when encoutering a zip file, try to unzip it first and then load other files.
10. **Use search only on very long documents and when identifying the exact keywords is needed**: Shorter documents, or the cases when exact match is not required, can be readed by read_text or read_screenshot directly.

## Strategies
1. **Step-by-Step Approach**: For long doc, break down to pages. Determine how many pages to load based on the number of tokens of each page.
2. **Reflection**: Regularly reflect on previous steps. If you encounter recurring errors despite multiple attempts, consider trying alternative methods.
3. **Review progress state**: Remember to review the progress state and compare previous information to the current web page to make decisions.
4. **See, Think and Act**: For each output, first provide a `Thought`, which includes a brief description of the current state and the rationale for your next step. Then generate the action `Code`.
5. Read a fair amount of pages each time. Make sure the total number tokens of the visible pages is less than MAX_FILE_READ_TOKENS.
6. Read a fair amount of screenshots each time. Make sure the total number of images is less than MAX_FILE_SCREENSHOT.

## Examples
Here are some example action outputs:

Thought: I need to first load the file.
Code:
```python
result=load_file("paper.pdf") 
print(result) 
```

Thought: Based on the meta information of the file, the number of tokens for each page is  Page 0: 100 , Page 1: 100 , page 2: 100. The total number of pages is 300, which is a fair amount of tokens to be directly processed by the LLM. So I can read all the pages.
Code:
```python
result=read_text("paper.pdf", [0, 1, 2]) # read the 0-th page of paper.pdf represented by markdown.
print(result)  # print the final result
```

Thought: It seems that the textual information alone is not sufficient for answering the query. Consider adding screenshot to the input. For example, if there is structured input in the text, like table, you should use read_screenshot.
Code:
```python
result=read_screenshot("paper.pdf", [0, 1, 2]) # read the 0, 1, 2 pages of paper.pdf represented by markdown.
print(result)  # print the final result
```

Thought: Based on the meta information of the file, the number of tokens for each page is  Page 0: 1000, Page 1: 1000, page 2: 1000, page 3: 1000, page 4: 1000, page 5: 1000, page 6: 1000. Since I should read less than 4000 tokens each time, I need to read page [0, 1, 2, 3] first.
Code:
```python
result=read_screenshot("paper.pdf", [0, 1, 2, 3]) # read 0, 1, 2, 3 pages, which has 4000 tokens that satisfies the constraint.
print(result)  # print the final result
```


Thought: There are 400 pages in the file and the task requires getting the page number of the page that contains "Geoffrey Hinton", so I need to use `search` function
Code:
```python
result=search("paper.pdf", ["Geoffrey Hinton"]) # search for the word "Geoffrey Hinton" and return the page numbers.
print(result)  # print the number of pages that contains the keyword "Geoffrey Hinton"
```

Thought: I have gathered all the information needed. Now, I need to use some python code to compute and stop with the final answer.
Code:
```python
# Extracted information about Rick Riordan's books and their status
all_status = [
    "Checked Out",  
    "Overdue",      
]

# Count the number of books that are not currently on the library's shelves
not_on_shelves_count = sum(status in ["Checked Out", "Overdue"] for status in all_status)
result=stop(answer=not_on_shelves_count, summary="The task is completed.")
print(result)
```

Thought: After trying many times of writing code to solve the problem but failed, I need to directly answer the question using my parametric knowledge. 
Code:
```python
result=stop(answer=my_answer, summary="Directly answer the question.")
print(result)
```

Thought: After failing many times or suffering from jailbreak or content filter, I cannot find any useful information
Code:
```python
result=stop(answer="", summary="Return an empty string")
print(result)
```
"""

_FILE_END_SYS = """You are a proficient assistant tasked with generating a well-formatted output for the execution of a specific task by an agent.

## Available Information
- `Target Task`: The specific task to be accomplished.
- `Recent Steps`: The latest actions taken by the agent.
- `Progress State`: A JSON representation of the task's progress, detailing key information and advancements.
- `Final Step`: The last action before the agent's execution concludes.
- `Stop Reason`: The reason for stopping. If the task is considered complete, this will be "Normal Ending".

## Guidelines
1. **Goal**: Deliver a well-formatted output. Adhere to any specific format if outlined in the task instructions.
2. **Code**: Generate a Python dictionary representing the final output. It should include two fields: `output` and `log`. The `output` field should contain the well-formatted final result, while the `log` field should summarize the navigation trajectory.
3. **Failure Mode**: If the task is incomplete (e.g., due to issues like "Max step exceeded"), the output should be an empty string. Provide detailed explanations and rationales in the log field, which can help the agent to better handle the target task in the next time. If there is partial information available, also record it in the logs.

## Examples
Here are some example outputs:

Thought: The task is completed with the requested information found.
Code:
```python
{
    "output": "The number of mentions of Geoffrey Hinton in the reference of the paper is 5.",  # provide a well-formatted output
    "log": "The task is completed. The result is found on the page ...",  # a summary of the navigation details
}
```

Thought: The task is incomplete due to "Max step exceeded",
Code:
```python
{
    "output": "",  # make it empty if no meaningful results
    "log": "The task is incomplete due to 'Max step exceeded'. The agent first navigates to the main page of ...",  # record more details in the log field
}
```
"""

def _prepare_imgs(image_suffix, visual_content):
    if isinstance(image_suffix, str):
        image_suffix = [image_suffix]
    if len(image_suffix) < len(visual_content):
        image_suffix = image_suffix + ["png"] * (len(visual_content) - len(image_suffix))
    ret = [{'type': 'image_url', 'image_url': {"url": f"data:image/{s};base64,{img}"} } for s, img in zip(image_suffix, visual_content)]
    return ret

def file_plan(**kwargs):
    user_lines = []
    user_lines.append(f"## Target Task\n{kwargs['task']}\n\n")  # task
    user_lines.append(f"## Recent Steps\n{kwargs['recent_steps_str']}\n\n")
    user_lines.append(f"## Previous Progress State\n{kwargs['state']}\n\n")
    user_lines.append(f"## Paths of all Files\n{kwargs['loaded_files']}\n\n")  # file information
    user_lines.append(f"## Meta information of all Files\n{kwargs['file_meta_data']}\n\n")  # file meta information like page number, token number
    user_lines.append(f"## Current Content\n{kwargs['textual_content']}\n\n")  # current visible information
    user_lines.append(f"## Target Task (Repeated)\n{kwargs['task']}\n\n")  # task
    
    user_lines.append("""## Output
Please generate your response, your reply should strictly follow the format:
Thought: {Provide an explanation for your planning in one line. Begin with a concise review of the previous steps to provide context. Next, describe any new observations or relevant information obtained since the last step. Finally, clearly explain your reasoning and the rationale behind your current output or decision.}
Code: {Then, output your python dict of the updated progress state. Remember to wrap the code with "```python ```" marks.}
""")
    user_str = "".join(user_lines)
    ret = [{"role": "system", "content": _FILE_PLAN_SYS}, {"role": "user", "content": user_str}]

    if kwargs["visual_content"] is not None:
        assert kwargs['image_suffix'] is not None
        ret[-1]['content'] = [{'type': 'text', 'text': ret[-1]['content'] + "\n\n## Screenshot of the current pages."}] + _prepare_imgs(kwargs['image_suffix'], kwargs['visual_content'])

    # ret[-1]['content'] = [{'type': 'text', 'text': ret[-1]['content'] + "\n\n## Screenshot of the current webpage."}, 
    #                         {'type': 'image_url', 'image_url': {"url": f"data:image/png;base64,{kwargs['screenshot']}"}},
    #                         ]
        
    return ret

def file_action(**kwargs):
    user_lines = []
    user_lines.append(f"## Target Task\n{kwargs['task']}\n\n")  # task
    user_lines.append(f"## Recent Steps\n{kwargs['recent_steps_str']}\n\n")
    user_lines.append(f"## Progress State\n{kwargs['state']}\n\n")
    user_lines.append(f"## Paths of all Files\n{kwargs['loaded_files']}\n\n")  # file information
    user_lines.append(f"## Meta information of all Files\n{kwargs['file_meta_data']}\n\n")  # file meta information like page number, token number
    user_lines.append(f"## Current Content\n{kwargs['textual_content']}\n\n")  # current visible information
    user_lines.append(f"## Target Task (Repeated)\n{kwargs['task']}\n\n")  # task
    user_lines.append("""## Output
Please generate your response, your reply should strictly follow the format:
Thought: {Provide an explanation for your action in one line. Begin with a concise review of the previous steps to provide context. Next, describe any new observations or relevant information obtained since the last step. Finally, clearly explain your reasoning and the rationale behind your current output or decision.}
Code: {Then, output your python code blob for the next action to execute. Remember that you should issue **ONLY ONE** action for the current step. Remember to wrap the code with "```python ```" marks.}
""")
    user_str = "".join(user_lines)
    ret = [{"role": "system", "content": _FILE_ACTION_SYS.replace("MAX_FILE_READ_TOKENS", str(kwargs['max_file_read_tokens'])).replace("MAX_FILE_SCREENSHOT", str(kwargs['max_file_screenshots']))}, 
           {"role": "user", "content": user_str}] 
    if kwargs["visual_content"] is not None:
        assert kwargs['image_suffix'] is not None
        ret[-1]['content'] = [{'type': 'text', 'text': ret[-1]['content'] + "\n\n## Screenshot of the current pages."}] + _prepare_imgs(kwargs['image_suffix'], kwargs['visual_content'])
    return ret

def file_end(**kwargs):
    user_lines = []
    user_lines.append(f"## Target Task\n{kwargs['task']}\n\n")  # task
    user_lines.append(f"## Recent Steps\n{kwargs['recent_steps_str']}\n\n")
    user_lines.append(f"## Progress State\n{kwargs['state']}\n\n")
    user_lines.append(f"## Current Content\n{kwargs['textual_content']}\n\n")  # task
    user_lines.append(f"## Final Step\n{kwargs['current_step_str']}\n\n")
    user_lines.append(f"## Stop Reason\n{kwargs['stop_reason']}\n\n")
    user_lines.append(f"## Target Task (Repeated)\n{kwargs['task']}\n\n")  # task
    user_lines.append("""## Output
Please generate your response, your reply should strictly follow the format:
Thought: {First, within one line, explain your reasoning for your outputs.}
Code: {Then, output your python dict of the final output. Remember to wrap the code with "```python ```" marks.}
""")
    user_str = "".join(user_lines)
    ret = [{"role": "system", "content": _FILE_END_SYS}, {"role": "user", "content": user_str}]
    return ret

# --
PROMPTS = {
"file_plan": file_plan,
"file_action": file_action,
"file_end": file_end,
}
