## A light-weight multi-agent collaboration framework for complex task-solving

## Running

### Environment

#### Python
- 目前没有放在docker里面，一般的python环境即可运行（建议使用python3.12）。
- 需要安装以下依赖包：
```bash
pip install boto3 botocore openai duckduckgo_search rich numpy openpyxl biopython mammoth markdownify pandas pdfminer-six python-pptx pdf2image puremagic pydub SpeechRecognition bs4 youtube-transcript-api requests transformers protobuf langchain langchain-openai
# for ck_web2
pip install selenium helium smolagents
```

#### Web

- **On Linux**:

  - web-agent需要启一个web-browser-server：[_web](./ck_web/_web/run_local.sh).
  - 需要安装以下依赖包：
  ```bash
  apt-get install -y poppler-utils default-jre libreoffice-common libreoffice-java-common libreoffice ffmpeg
  # for ck_web
  sh ckv3/ck_web/_web/run_local.sh
  ```
  - **IMPORTANT**: it is recommended to run this program in a sandbox since the generated python code is directly executed and currently there are no safety checkings. (Disable sudo for your user to ensure safety.)
  ```bash
  # run with root
  echo "${USER}" 'ALL=(ALL) NOPASSWD: !ALL' | tee /etc/sudoers.d/${USER}-rule
  chmod 440 /etc/sudoers.d/${USER}-rule
  deluser ${USER} sudo
  hostnamectl set-hostname localhost
  ```
- **On Mac**:
  - web-agent需要启一个web-browser-server：[_web](./ck_web/_web/run_local_mac.sh).
  - 需要安装以下依赖包：
  ```zsh
  brew install --cask libreoffice
  brew install poppler
  brew install ffmpeg
  # for ck_web
  sh ckv3/ck_web/_web/run_local_mac.sh
  ```
  - **IMPORTANT**: it is recommended to run this program in a sandbox since the generated python code is directly executed and currently there are no safety checkings. (Disable sudo for your user to ensure safety.)
  ```bash
  # run with root
  echo "${USER}" 'ALL=(ALL) NOPASSWD: !ALL' | tee /etc/sudoers.d/${USER}-rule
  chmod 440 /etc/sudoers.d/${USER}-rule
  dseditgroup -o edit -d "$USER" admin
  scutil --set HostName localhost
  ```

### Running Configuration
通过一个整体的configuration-python-dictionary来指定运行参数，字典里的值直接对应类里面的item（支持hierarchy的方式）。init具体的mechanism详见`utils.py:KwargsInitializable`，简单来说是通过赋值`__dict__`来直接修改object。以下是对于`CKAgent`的一个例子：
```python
{
    "model": {"call_target": "gpt:gpt-4o-mini"},  # use gpt-4o-mini for the LLM of main agent
    "max_steps": 10,  # a maximum of 10 steps for the main agent
    "web_agent": {
        "model": {"call_target": "http://MY_VLLM:8080/v1/chat/completions"},  # use vllm service (replace MY_VLLM with your IP) for the web agent
        "web_env_kwargs": {"web_ip": "localhost:3000"},  # IP for the web-browser server
    }
}
```

### Example (simple-test)
- See [`ck_main/_test`](./ck_main/_test) for a simple example and its corresponding outputs
```bash
export PYTHONPATH=/your/path/to/cognitive_kernel_GAIA//
# Assume we have set up a vllm model server and a web-browser server (currently these are active: WEB_IP: a somewhat constrained env, LLM_URL: qwen2.5-72B-instruct)
#WEB_IP=<YOUR_WEB_BROWSER_SERVER_IP>:<PORT>  # web-browser server
WEB_IP=localhost:3001  # web-browser server
LLM_URL=http://<YOUR_VLLM_HOST>:8080/v1/chat/completions  # vllm model server
#LLM_URL=gpt:gpt-4.1  # using gpt
#VLM_URL=gpt:gpt-4.1  # using gpt
#LLM_URL=claude:  # using claude
#VLM_URL=claude:  # using claude
# run simple test
MAIN_ARGS="{'web_agent': {'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}, 'web_env_kwargs': {'web_ip': '${WEB_IP}'}}, 'file_agent': {'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}}, 'model': {'call_target': '${LLM_URL}'}}"
# use "NO_NULL_STDIN=1" for easier debugging
# you can also remove `--input` field to directly input your task from stdin
# you can also remove `-mpdb` flag to run the program directly instead of in debugging mode
NO_NULL_STDIN=1 python3 -u -mpdb -m ckv3.ck_main.main --updates "${MAIN_ARGS}" --input /your/path/to/simple_test.jsonl --output /your/path/to/simple_test.output.jsonl |& tee _log_simple_test
less -R _log_simple_test  # use 'less -R' to see the colored outputs
```

### Example (test-gaia)
```bash
# Step 1: prepare data
# decompress the gaia data (or you can download it by yourself from huggingface)
# -> assume all the gaia related input files are at the same DIR as the input json meta-file
unzip /your/path/to/cognitive_kernel_GAIA/Evaluation/gaia2504.zip
# Step 2: prepare web service (recommending using a PC or laptop to enable better network connection)
# -> prepare things according to "./ck_web/_web/run_local.sh"
#LISTEN_PORT=3001 npm start
#WEB_IP=localhost:3001  # web-browser server
# Step 3: prepare a vllm instance for model calling
# use gpt
#LLM_URL=gpt:gpt-4.1
#VLM_URL=gpt:gpt-4.1
#export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT"
#export AZURE_OPENAI_API_KEY="YOUR_API_KEY"
#export AZURE_OPENAI_API_VERSION="YOUR_API_VERSION"
# or use claude
#LLM_URL=claude:  # using claude
#VLM_URL=claude:  # using claude
#export AWS_ACCESS_KEY="YOUR_KEY"
#export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
# or simply use our model
# VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve _run_orig/Qwen2.5-72B-Instruct/ --tensor-parallel-size 8 --served-model-name ck --port 8080 --enforce-eager &
# VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve _run_orig/Qwen2.5-72B-Instruct/ --tensor-parallel-size 4 --served-model-name ck --port 8080 --enforce-eager &
# VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve _run_orig/Qwen2.5-VL-72B-Instruct/ --tensor-parallel-size 4 --served-model-name ck --port 8081 --enforce-eager &

export PLAYWRIGHT_BACKEND=browserless # or local
export BROWSERLESS_TOKEN=YOUR_TOKEN # browserless token

LLM_URL=http://<YOUR_VLLM_HOST>:8080/v1/chat/completions  # vllm model server
VLM_URL=http://<YOUR_VLLM_HOST>:8081/v1/chat/completions  # for VLM
# curl ${LLM_URL} -H "Content-Type: application/json" -d '{"model": "ck", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who won the world series in 2020?"}]}'
# Step 4: Setup search engine
# either using google api
#export SEARCH_BACKEND="Google"
#export SEARCH_API_KEY="YOUR_API_KEY"
#export SEARCH_CSE_ID="YOUR_CSE_ID"
# or simply use DuckDuckGo
export SEARCH_BACKEND="DuckDuckGo"
# Step 5: run
export PYTHONPATH=/your/path/to/cognitive_kernel_GAIA/
#pip install ...  # see above in `Environment`
# it will be more stable to run a new web-browser for each web call, setup WEB_PORT (web browser service's port) and WEB_DIR (main dir of the web browser service)
# moreover, it is slightly better to use non-boxed screenshot (make sure to update the latest `server.js` and set screenshot_boxed=False)
WEB_DIR=/mnt/d/zh/data/ck/_web/  # where we put `server.js` and related `node_modules`
WEB_PORT=3001
MAIN_ARGS="{'web_agent': {'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}, 'web_env_kwargs': {'web_ip': 'localhost:${WEB_PORT}', 'web_command': 'cd ${WEB_DIR}; LISTEN_PORT=${WEB_PORT} npm start', 'screenshot_boxed': False}}, 'file_agent': {'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}}, 'model': {'call_target': '${LLM_URL}'}}"
python3.12 -u -m ckv3.ck_main.main --updates "${MAIN_ARGS}" --input /your/path/to/gaia_dev.jsonl --output /your/path/to/gaia_dev.output.jsonl |& tee -a _log_gaia_dev
# "{'step_mrun': 5, 'mrun_multimodal_count': 2}"  # multiple runs for web-agent (3 llm + 2 vlm)
# Step 6: analyze and check the output
python -m ckv3.ck_main.scripts.analyze -f /your/path/to/output/gaia_dev.output.jsonl -b 0
```

### Extra Running Config
```bash
# calling claude+thinking for the outside main-agent
LLM_URL=gpt:gpt-4.1  # still use gpt4.1 for sub-agents
VLM_URL=gpt:gpt-4.1
export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT"  # find these keys in the corresponding spreadsheets
export AZURE_OPENAI_API_KEY="YOUR_API_KEY"
export AZURE_OPENAI_API_VERSION="YOUR_API_VERSION"
export AWS_ACCESS_KEY="YOUR_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
MAIN_ARGS="{'web_agent': {'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}, 'web_env_kwargs': {'web_ip': 'localhost:${WEB_PORT}', 'web_command': 'cd ${WEB_DIR}; LISTEN_PORT=${WEB_PORT} npm start', 'screenshot_boxed': False}}, 'file_agent': {'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}}, 'model': {'thinking': 'True', 'call_target': 'claude:', 'call_kwargs': {'temperature': 0.2, 'top_p': 0.95, 'max_tokens': 4096}}}"  # use claude+thinking for main-agent, allowing more max_token budgets
```

## Data

### Saved Data Format

目前数据格式如下：
- 其中session对于Session类，用于保存整体trajectory [session.py](./agents/session.py)
- 这里的analysis脚本对于理解数据格式可能有所帮助 [analyze.py](./ck_main/scripts/analyze.py)
```python
# one instance in one json-line
INSTANCE = {
  "id": "Task ID",
  "task": "Task Description",
  "session": {  # corresponding to the class of Session
    "id": "Session ID",
    "info": {...},  # other information such model calling token counts
    "task": "Original Task Description",
    "steps": [  # information for each step
      {
        "step_idx": 0,
        "plan": {
          "thought": "Model's thought",
          "code": "Model's output code",
          "state": {...},  # updated state
          "llm_input": [],  # model's direct input messages
          "llm_output": "Model's raw output",  # model's raw output
        },
        "action": {
          "...": ...,  # similar to plan
          # "observation": ...,  # simple outputs from code execution
          # if calling a sub-agent, we have more complex structures storing the session from the sub-agent
          "observation": {  # see the class of AgentResult
            "output": "formatted outputs",
            "log": "logs",
            "task": "Task for the sub-agent",
            "session": {...},
          },
        },
      },  # step 0
      ...,  # later steps
      {
        "...": ...,  # plan and action
        "end": {  # in the final step, we may also have an ending module if configured
          "..."  # fields are similar to plan and action
        }
      }  # final step
    ],
  },
}
```

### LLM-Calling Input

目前prompt/llm-call的部分采用的是prompt拼接的方式，而不是通常的multi-turn-dialog的形式，即llm-call的input是只有两轮的messages: `[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]`。这里`SYSTEM_PROMPT`指定当前模块general的一些rules，`USER_PROMPT`则主要表达当前specific-task以及状态信息，以下是一个例子：
```json
[
  {"role": "system", "content": "You are a helpful assistant to ... You have the following actions to choose: ... You have the follow tools to use ..."}, 
  {"role": "user", "content": "Here is the task you are going to solve ... Here are the previous steps ... "}
]
```
具体prompt的形式可以查看具体的`prompts.py`文件，例如[ck_main/prompts.py](./ck_main/prompts.py)，[ck_web/prompts.py](./ck_web/prompts.py).

### LLM-Calling Output

大部分模型以代码作为action，输出也以code作为主要结果，基本的格式为：
```text
Thought: {First, within one line, explain your reasoning for your action.}
Code: {Then, output your python code blob for the next action to execute. Remember to wrap the code with "```python ```" marks.}
```
例如：
```text
Thought: The task is completed with the requested price found.
Code:
```python
{
    "output": "The price of the iphone 16 is $799.",  # provide a well-formatted output
    "log": "The task is completed. The result is found on the page ...",  # a summary of the navigation details
}
```

## Detailed Notes

### Main Principles
- multi-agent collaboration，即main-agent可以类似函数调用的方式使用sub-agent，目的是可能可以简化每个agent需要处理问题的复杂度，另外也可以简化训练数据收集。（如果这个方式不够有效的话也可以fallback到vanilla的单一main-agent）
- planning by state tracking，可以选择使用planning模块来保存一个中间状态来存储相关信息（使用一个python-dictionary），planning即根据当前环境信息来对状态进行update。（planning比较类似cot的thinking过程，可以根据module的复杂程度选择on/off）

### Source
- 整体framework类似huggingface-smolagents，不过做了很大部分的简化处理。
- web-browser以及code-executor部分主要源自于CK-V2。
- Prompt部分的处理（数据input/output）主要来源于之前的rollback-project。

### Detailed Notes
- ckv3.agents:
  - 整体框架类似huggingface-smolagents，做了很大部分的简化处理。
  - `utils.py`: 一些helper-function，这里有一些helper-class可以注意一下：
    - `KwargsInitializable`: 采用了一种非常规的简化configuration的方式，即从__init__的**kwargs中读取外部传入的参数来做configuration。这里有一点tricky需要注意的是对于子类来说，也需要通过super()的方式来做init，详情参考`ck_web/agent.py:WebAgent`的init作为一个例子。
    - `TemplatedString`: 简化prompt的定义形式，可以使用一个函数或者fstring来定义prompt-template，详情参考`{ck_web,ck_main}/prompts.py`。
    - `CodeExecutor`: 还是类似之前CK系统直接使用exec来进行python代码执行（简单直接一点）。
  - `session.py`: 定义主要的`AgentSession`类:
    - 用于储存一个task-solving中的相关信息，其中`AgentSession.steps`储存每一步的信息。
  - `model.py`: 定义主要的`LLM`类，其中：
    - `LLM.call_target`: 如果指定为"manual"，则代表用户输入可以方便debug，如果是"gpt:{gpt_model_name}"，则是调用gpt_model_name（详见`OpenaiHelper`类），如果是"http"开头，则是代表使用remote部署好的vllm-service。
    - `LLM.call_kwargs`指定默认的LLM-call的参数。
    - `LLM.__call__`在调用的外面加了一个retry的wrapper，即如果发生了错误会进行retry（次数由`LLM.max_retry_times`指定）。这个LLM-call的input/output格式在Data-Section有详细描述。
  - `tool.py`: 定义主要的`Tool`类，其中：
    - `Tool`类做了非常大的简化，需要定义特定的实现函数（用于实际代码调用），以及函数定义（用于prompt-input）。
    - `StopTool`: 一个特殊的函数stop来标记任务结束。
  - `agent.py`: 定义主要的`MultiStepAgent`类，其中：
    - `MultiStepAgent.sub_agents`和`MultiStepAgent.tools`是agent可以使用的子函数，sub_agent代表子模块也是一个(LLM-based) agent，tool则直接代表一个pre-defined的python-function（定义于Tool类）。
    - `MultiStepAgent.model`是一个`model.py:LLM`，处理这个agent的llm-call的实际调用情况。
    - `MultiStepAgent.templates`储存不同module的prompt-template，这些template可以使用`register_template`/`get_template`来进行定义和读取。
    - `MultiStepAgent.max_steps`表示当前agent最大执行步骤数，`MultiStepAgent.recent_steps`代表最近多少步骤的信息会加在input-prompt里面，`MultiStepAgent.store_io`表示是否储存每次LLM调用的input/output信息（储存文件会很大，但可以比如用于训练），`MultiStepAgent.active_functions`表示哪些sub-agent以及tool是active的（加入input-prompt）。
    - `MultiStepAgent.__call__`和`MultiStepAgent.get_function_definition`用于agent作为sub-agent被其他agent调用时候的情况，`get_function_definition`返回函数definition-line放在input-prompt里面，`__call__`的protocol则统一为：输入task（任务instruction）；输出output（指定格式的输出），log（输出之外的其他信息，例如错误信息）。
    - `MultiStepAgent.run`和`MultiStepAgent.yield_session_run`: 主要的running-loop，初始化`AgentSession`来储存整个procedure的信息，`progress_state`来表示solving的状态，每一步使用`MultiStepAgent.step`来进行一步的操作；最后使用`MultiStepAgent.finalize`来format最终输出的结果。
    - `MultiStepAgent.step`: 每一个步骤中，如果指定了plan的template，则执行plan模块操作来更新`progress_state`，之后执行action模块，得到当前action-code，调用`MultiStepAgent.step_action`来执行action操作（by-default用code-executor来跑生成的代码，一些特殊的类可能还有额外的操作）。每次LLM调用的时候，首先准备好相应的input_kwargs（`MultiStepAgent._prepare_common_input_kwargs`），之后使用`self.templates["module_name"].format(**_input_kwargs)`来得到LLM调用的输入（详见下面Data-Section的Input格式），LLM调用(`MultiStepAgent._call_model`)返回是一个str（详见下面Data-Section的Output格式）可以使用`MultiStepAgent._parse_output`来parse。
    - `MultiStepAgent.finalize`: 用于最终结果的format，也可以指定LLM以及相应的template来完成。
    - To be implemented methods in sub-classes: `init_run`(每个run开始之前的一些处理), `end_run`(每个run结束之后的一些处理), `step_prepare`(每一步的prompt的input-kwargs准备), `step_action`(每一步的action-execution), `step_check_end`(当前步骤之后是否结束这个run)。
- ckv3.ck_web: web-agent
  - `_web`: 这里放了adapt过的web-browser-server(mainly from CK-v2)，这里面做了一些小的修改（例如加了以西try-catch以及加了goto方法），而且目前暂时把screenshot信息也disable掉了，之后也需要加回来。
  - `utils.py`: 主要定义helper类`WebEnv`以及相关状态`WebState`（based on call_web.py of CK-v2）。
  - `agent.py`: 定义`MultiStepAgent`的sub-class`WebAgent`，如`KwargsInitializable`所提到，这里的super()需要一些比较tricky的处理来正确初始化。
    - `WebAgent.web_envs`用于储存每个任务的`WebEnv`。
    - `PREDEFINED_WEB_ACTIONS`，对于code以及action-execution，目前这边的处理是让代码生成一个action-str(as defined in `PREDEFINED_WEB_ACTIONS`)，然后再到`WebEnv`再parse一遍（因为之前`WebEnv`已经写好了parse的函数就直接用了）。
    - init_run、end_run、step_prepare、step_action、step_check_end也针对web-env做了一些额外的操作。
  - `prompts.py`: web-agent的prompt-template，这里有三个模块：plan/action/end。
  - `main.py`：直接测试web-agent。
- ckv3.ck_web: main-agent
  - `agent.py`: 定义`MultiStepAgent`的sub-class`CKAgent`，如`KwargsInitializable`所提到，这里的super()需要一些比较tricky的处理来正确初始化。
    - 这里主要需要注意的是对于sub-agent和tool的添加，目前添加了一个web-agent和stop-tool，后续有新的sub-agent和tool也可以类似添加。
  - `prompts.py`: main-agent的prompt-template，这里简化了一下没有定义end模块，而是直接指定一个stop-tool来标记一个run的结束。
  - `main.py`：直接测试main-agent。

## Updates
- (25.06.17) Fix compatability for Qwen3-32B. Add maximum token truncation, fix format errors, update rejection sampling in convert_sft.py
- (25.04.10) Adjust prompts for the web-agent. (cognitive_kernel_v3:214e55879c799f0952e3ffbeda260dd0c53e5cdd)
- (25.04.11) Add an ending module with GAIA output format rules for main-agent. (cognitive_kernel_v3:4168506d5dbac9de384d593572b2c4081a99f28b)
- (25.04.11) Add multimodal for web-agent. (cognitive_kernel_v3:d8af25a6bbf490390b2a524819e4b02507c55375)
- (25.04.11) Add tools of ask_llm and simple_web_search. (bc0da08cb9ff5185b29993eef720ee7426403430)
- (25.04.15) Fix some bugs and add readme for gaia running. (e2f6e8f2ae8ace9af14fb50054a0ecdcaf10bd68)
- (25.04.16) Fix more bugs and add google-search. (4156431106081e58238f107e8270eceec181eb25)
- (25.04.17) Add support for step-wise multiple running with model-based voting. (fdad2d565058c66fe405bec0445302983b057e1c)
- (25.04.24) Fix prompts for better output format and avoid syntax problems. (f9c7e2aeab956e3bed879913e97056bdb95afbd8)
- (25.04.27) Add markdown repr for web agent. (185729bd6cfb392edf0002cd8ccd585ad7d89fd4)
- (25.04.30) Initial merging of file agent. (5f63fcbe/2e5fc952)
- (25.05.01) Specify explicit ActionResult for sub-agents built-in actions & Prompt adjustments. (43c4495e/21b37a99/56db7a37/ea4c0023)
- (25.05.07) Add configurable adaptive screenshot modes for web-agent. (4b9cfd09)
- (25.05.12) Some improvements if using individual web browser server and non-boxed screenshots. ()

## TODO
- [x] 如何结合web-agent以及search-engine，web-agent下载文件
- [x] web-agent的multimodal支持
- [x] pdf-agent/image-agent支持+其他文件
- [] parallel sub-agent running and voting?
- [] 外层main-ck-agent的workflow的具体design(how to handle errors?)
- [] 其他CK模块（例如memory, persona-profile）的enhancement
- [] front-end/docker/...的adaptation
- [] python-execution safety
- [] Misc: 1) web-browser cannot navigate to a web file (500 error), 2) cookies for web browser,
