import time
import requests
from .utils import wrapped_trying, rprint, GET_ENV_VAR, KwargsInitializable

from transformers import AutoTokenizer

class MessageTruncator:
    def __init__(self, model_name="Qwen/Qwen3-32B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _count_text_tokens(self, content):
        """
        Count tokens in a message's content.
        Handles both string and list-of-dict (multimodal) content.
        """
        if isinstance(content, str):
            return len(self.tokenizer.encode(content, add_special_tokens=False))
        elif isinstance(content, list):
            total = 0
            for part in content:
                if part.get("type") == "text":
                    total += len(self.tokenizer.encode(part.get("text", ""), add_special_tokens=False))
            return total
        else:
            return 0

    def _truncate_text_content(self, content, max_tokens):
        """
        Truncate text in content to fit max_tokens.
        For multimodal, only truncate the last text part if needed.
        """
        if isinstance(content, str):
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        elif isinstance(content, list):
            # Go through parts, keep images, truncate last text if needed
            new_content = []
            tokens_used = 0
            for i, part in enumerate(content):
                if part.get("type") == "text":
                    text = part.get("text", "")
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    if tokens_used + len(tokens) > max_tokens:
                        # Truncate this text part
                        remaining = max_tokens - tokens_used
                        truncated_tokens = tokens[:remaining]
                        truncated_text = self.tokenizer.decode(truncated_tokens)
                        if truncated_text:
                            new_content.append({"type": "text", "text": truncated_text})
                        break  # No more tokens allowed after this
                    else:
                        new_content.append(part)
                        tokens_used += len(tokens)
                else:
                    # Always keep images
                    new_content.append(part)
            return new_content
        else:
            return content

    def truncate_message_list(self, messages, max_length):
        """
        Truncate a list of messages so that the total token count does not exceed max_length.
        Keeps the most recent messages. If the most recent message alone exceeds max_length,
        its text content will be truncated to fit. Images are never truncated or removed.
        """
        truncated = []
        total_tokens = 0
        for msg in reversed(messages):
            content = msg.get("content", "")
            tokens = self._count_text_tokens(content)
            if total_tokens + tokens > max_length:
                if not truncated:
                    # Truncate the most recent message's text content to fit max_length
                    truncated_content = self._truncate_text_content(content, max_length)
                    truncated_msg = msg.copy()
                    truncated_msg["content"] = truncated_content
                    truncated.insert(0, truncated_msg)
                break
            truncated.insert(0, msg)
            total_tokens += tokens
        return truncated

# --
# helper
def update_stat(stat, call_return):
    usage = call_return.get("usage", {})
    if stat is not None:
        stat["llm_call"] = stat.get("llm_call", 0) + 1
        for k in ['completion_tokens', 'prompt_tokens', 'total_tokens']:
            k = {'outputTokens': 'completion_tokens', 'inputTokens': 'prompt_tokens', 'totalTokens': 'total_tokens'}.get(k, k) # handling keys in claude
            stat[k] = stat.get(k, 0) + usage.get(k, 0)
# --

class OpenaiHelper:
    _openai_clients = {}  # model_name -> Helper

    @staticmethod
    def get_openai_client(model_name="", api_endpoint="", api_key=""):
        model_name_suffix = f"_{model_name}" if model_name else ""
        
        is_databricks_model = "databricks" in model_name.lower()
        
        # Explicitly check for gpt-5-pro to route it away from Azure
        is_gpt5_pro_model = model_name == "gpt-5-pro"

        
        should_use_azure = not is_databricks_model and \
                           not is_gpt5_pro_model and \
                           GET_ENV_VAR("AZURE_OPENAI_API_KEY", f"AZURE_OPENAI_API_KEY{model_name_suffix}")
        
        if model_name not in OpenaiHelper._openai_clients:  # lazy init
            import openai
            
            if should_use_azure:
                client = openai.AzureOpenAI(
                    azure_endpoint=GET_ENV_VAR("AZURE_OPENAI_ENDPOINT", f"AZURE_OPENAI_ENDPOINT{model_name_suffix}", df=api_endpoint),
                    api_key=GET_ENV_VAR("AZURE_OPENAI_API_KEY", f"AZURE_OPENAI_API_KEY{model_name_suffix}", df=api_key),
                    api_version=GET_ENV_VAR("AZURE_OPENAI_API_VERSION", df="2024-02-01")
                )
            else:
                endpoint_string = "/openai/v1/"
                if is_databricks_model:
                    endpoint_string = ""
                client = openai.OpenAI(
                    base_url=GET_ENV_VAR("OPENAI_ENDPOINT") + endpoint_string,
                    api_key=GET_ENV_VAR("OPENAI_API_KEY", df=api_key),
                    default_query={"api-version": "preview"}, 
                )
            OpenaiHelper._openai_clients[model_name] = client
        return OpenaiHelper._openai_clients[model_name]

    @staticmethod
    def call_chat(messages, stat=None, **openai_kwargs):
        rprint(f"Call gpt with openai_kwargs={openai_kwargs}")
        
        
        _final_kwargs = openai_kwargs.copy()
        model_name = _final_kwargs.get("model", "")
        _client = OpenaiHelper.get_openai_client(model_name)
        
        
        
        if model_name == "GPT-5":
            if 'max_tokens' in _final_kwargs:
                _final_kwargs['max_completion_tokens'] = _final_kwargs.pop('max_tokens')
            _final_kwargs['temperature'] = 1.0
            if 'top_p' in _final_kwargs:
                del _final_kwargs['top_p']
            rprint(f"Adjusted kwargs for gpt-5: {_final_kwargs}")
        if 'databricks' in model_name:
            if 'top_p' in _final_kwargs:
                del _final_kwargs['top_p']
        if model_name == "gpt-5-pro":
            response_obj = _client.responses.create(
                model=model_name,
                input=messages,
                reasoning={
                    "effort": "high",
                    "summary": "detailed"
                },
            )

            response = response_obj.output_text
            
        else:
            
            chat_completion = _client.chat.completions.create(messages=messages, **_final_kwargs)
            call_return = chat_completion.to_dict()
            update_stat(stat, call_return)
            
            if "content" not in call_return["choices"][0]["message"] or call_return["choices"][0]["message"]["content"] is None:
                response = ""
            else:
                response = call_return["choices"][0]["message"]["content"]

      
        
        if response.strip() == "":
            error_details = locals().get('call_return', 'N/A for gpt-5-pro')
            raise RuntimeError(f"Get empty response from gpt: {error_details}")
        return response

class AzureClaudeHelper:
    _azure_claude_clients = {}  # model_name -> Helper

    @staticmethod
    def get_azure_claude_client(model_name="", api_endpoint="", api_key=""):
        model_name_suffix = f"_{model_name}" if model_name else ""
        if model_name not in AzureClaudeHelper._azure_claude_clients:  # lazy init
            from openai import OpenAI
            # Use the Azure Databricks endpoint format for Claude
            client = OpenAI(
                api_key=GET_ENV_VAR("DATABRICKS_TOKEN", f"DATABRICKS_TOKEN{model_name_suffix}", df=api_key),
                base_url=GET_ENV_VAR("DATABRICKS_BASE_URL", f"DATABRICKS_BASE_URL{model_name_suffix}", df=api_endpoint)
            )
            AzureClaudeHelper._azure_claude_clients[model_name] = client
        return AzureClaudeHelper._azure_claude_clients[model_name]

    @staticmethod
    def call_chat(messages, stat=None, **azure_claude_kwargs):
        rprint(f"Call Azure Claude with azure_claude_kwargs={azure_claude_kwargs}")
        _client = AzureClaudeHelper.get_azure_claude_client(azure_claude_kwargs["model"])
        chat_completion = _client.chat.completions.create(messages=messages, **azure_claude_kwargs)
        call_return = chat_completion.to_dict()
        update_stat(stat, call_return)
        if "content" not in call_return["choices"][0]["message"]:
            response = ""
        else:
            response = call_return["choices"][0]["message"]["content"]
        if response.strip() == "":
            raise RuntimeError(f"Get empty response from Azure Claude: {call_return}")
        return response
    
class Boto3Helper:
    _boto3_client = {}  # model_name -> Helper

    @staticmethod
    def get_boto3_client(model_name="", region_name = 'us-west-2', api_key="", api_secret_key=""):
        model_name_suffix = f"_{model_name}" if model_name else ""
        if model_name not in Boto3Helper._boto3_client:  # lazy init
            import boto3
            if GET_ENV_VAR("AWS_ACCESS_KEY", f"AWS_ACCESS_KEY{model_name_suffix}") and GET_ENV_VAR("AWS_SECRET_ACCESS_KEY", f"AWS_SECRET_ACCESS_KEY{model_name_suffix}"):
                client = boto3.client("bedrock-runtime", 
                      region_name=GET_ENV_VAR("AWS_REGION_NAME", df=region_name),
                      aws_access_key_id=GET_ENV_VAR("AWS_ACCESS_KEY", f"AWS_ACCESS_KEY{model_name_suffix}", df=api_key),
                    aws_secret_access_key=GET_ENV_VAR("AWS_SECRET_ACCESS_KEY", f"AWS_SECRET_ACCESS_KEY{model_name_suffix}", df=api_secret_key))
            else:
                raise NotImplementedError
            Boto3Helper._boto3_client[model_name] = client
        return Boto3Helper._boto3_client[model_name]
    
    @staticmethod
    def to_bedrock_messages(messages):
        
        def _to_bedrock_message(message):
            import base64
            if isinstance(message["content"], str):
                return [{"text": message["content"]} ]
            else:
                new_message = []
                for item in message["content"]:
                    _type = item['type']
                    if _type == "text":
                        new_message.append({"text": item["text"]})
                    elif _type == "image_url":
                        # 'data:image/{image_suffix};base64'
                        import re
                        pattern = r'data:image/(?P<suffix>[^;]+);base64'
                        match = re.search(pattern, item['image_url']['url'])
                        
                        if match:
                            image_suffix = match.group('suffix')
                        else:
                            raise ValueError("Invalid image URL format")
                        new_message.append({"image": {
                                'format': image_suffix if image_suffix in ['png', 'jpeg', 'webp'] else 'png',
                                "source": {
                                    "bytes": base64.b64decode(item['image_url']['url'][len(f'data:image/{image_suffix};base64,'):] )
                                }
                            }})
                return new_message

        return [
            {"role": message["role"].replace("system", "user"), "content": _to_bedrock_message(message) }
            for message in messages
        ]

    @staticmethod
    def call_chat(messages, stat=None, **boto3_kwargs):
        rprint(f"Call gpt with boto3_kwargs={boto3_kwargs}")
        _client = Boto3Helper.get_boto3_client(boto3_kwargs["model"])
        messages = Boto3Helper.to_bedrock_messages(messages)
        if boto3_kwargs['thinking']:
            reasoning_config = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 2000
                }
            }
            chat_completion = _client.converse(modelId=GET_ENV_VAR("AWS_MODEL_ID", df="us.anthropic.claude-3-7-sonnet-20250219-v1:0"), messages=messages, additionalModelRequestFields=reasoning_config)
        else:
            chat_completion = _client.converse(modelId=GET_ENV_VAR("AWS_MODEL_ID", df="us.anthropic.claude-3-7-sonnet-20250219-v1:0"), messages=messages)
        
        call_return = chat_completion
        update_stat(stat, call_return)

        content_blocks = call_return["output"]["message"]["content"]

        reasoning = None
        text = None

        for block in content_blocks:
            if "reasoningContent" in block:
                reasoning = block["reasoningContent"]["reasoningText"]["text"]
            if "text" in block:
                response = block["text"]
        if reasoning is not None:
            response = f"Reasoning: {reasoning}\n{response}"
            
        if response.strip() == "":
            raise RuntimeError(f"Get empty response from claude: {call_return}")
        return response


class LLM(KwargsInitializable):
    def __init__(self, **kwargs):
        # basics
        self.call_target = "manual"  # fake=fake, manual=input, gpt(gpt:model_name)=openai [such as gpt:gpt-4o-mini], request(http...)=request
        self.thinking = False
        self.print_call_in = "white on blue"  # easier to read
        self.print_call_out = "white on green"  # easier to read
        self.max_retry_times = 5  # <0 means always trying
        self.seed = 1377  # zero means no seed!
        # request
        self.request_timeout = 100  # timeout time
        self.max_token_num = 20000
        self.call_kwargs = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 4096}  # other kwargs for gpt/request calling
        # --
        super().__init__(**kwargs)  # init
        # --
        # post init
        self.call_target_type = self.get_call_target_type()
        self.call_stat = {}  # stat of calling
        # --
        self.message_truncator = MessageTruncator()

    def __repr__(self):
        return f"LLM(target={self.call_target},kwargs={self.call_kwargs})"

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed

    def __call__(self, messages, **kwargs):
        func = lambda: self._call_with_messages(messages, **kwargs)
        return wrapped_trying(func, max_times=self.max_retry_times)

    def get_call_stat(self, clear=False):
        ret = self.call_stat.copy()
        if clear:  # clear stat
            self.clear_call_stat()
        return ret

    def clear_call_stat(self):
        self.call_stat.clear()

    def get_call_target_type(self):
        _trg = self.call_target
        if _trg == "manual":
            return "manual"
        elif _trg == "fake":
            return "fake"
        elif _trg.startswith("gpt:"):
            return "gpt"
        elif _trg.startswith("http"):
            return "request"
        elif _trg.startswith("claude:"):
            return "claude"
        elif _trg.startswith("azure_claude:"):
            return "azure_claude"
        else:
            raise RuntimeError(f"UNK call_target = {_trg}")

    def show_messages_str(self, messages, calling_kwargs, rprint_style):
        ret_ss = []
        if isinstance(messages, list):
            for one_mesg in messages:
                _content = one_mesg['content']
                if isinstance(_content, list):
                    _content = "\n\n".join([(z['text'] if z['type']=='text' else f"<{str(z)[:150]}...>") for z in _content])
                ret_ss.extend([f"=====\n", (f"{one_mesg['role']}: {_content}\n", rprint_style)])
        else:
            ret_ss.append((f"{messages}\n", rprint_style))
        ret = [f"### ----- Call {self.call_target} with {calling_kwargs} [ctime={time.ctime()}]\n{'#'*10}\n"] + ret_ss + [f"{'#'*10}"]
        return ret

    # still return a str here, for simplicity!
    def _call_with_messages(self, messages, **kwargs):
        time0 = time.perf_counter()
        _call_target_type = self.call_target_type
        _call_kwargs = self.call_kwargs.copy()
        _call_kwargs.update(kwargs)  # this time's kwargs
        if self.print_call_in:
            rprint(self.show_messages_str(messages, _call_kwargs, self.print_call_in))  # print it out
        # --
        if _call_target_type == "manual":
            user_input = input("Put your input >> ")
            response = user_input.strip()
            ret = response
        elif _call_target_type == "fake":
            ret = "You are correct! As long as you are happy!"
        elif _call_target_type == "gpt":
            ret = self._call_openai_chat(messages, **_call_kwargs)
        elif _call_target_type == "claude":
            _call_kwargs['thinking'] = self.thinking or self.thinking == "True"
            ret = self._call_claude_chat(messages, **_call_kwargs)
        elif _call_target_type == "azure_claude":
            ret = self._call_azure_claude_chat(messages, **_call_kwargs)
        elif _call_target_type == "request":
            messages = self.message_truncator.truncate_message_list(messages, self.max_token_num)
            headers = {"Content-Type": "application/json"}
            if isinstance(messages, list):
                json_data = {
                    "model": "ck",
                    "stop": ["<|eot_id|>", "<|eom_id|>", "<|im_end|>"],
                    "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
                if self.seed != 0:  # only if non-zero!
                    json_data.update(seed=self.seed)
            else:  # directly put it!
                json_data = messages.copy()
            json_data.update(_call_kwargs)
            r = requests.post(self.call_target, headers=headers, json=json_data, timeout=self.request_timeout)
            assert (200 <= r.status_code <= 300), f"response error: {r.status_code} {json_data}"
            call_return = r.json()
            if isinstance(call_return, dict) and "choices" in call_return:
                update_stat(self.call_stat, call_return)
                ret0 = call_return["choices"][0]
                if "message" in ret0:
                    ret = ret0["message"]["content"]  # chat-format
                    import re
                    ret = re.sub(r'<think>.*?</think>', '', ret, flags=re.DOTALL)
                else:
                    ret = ret0["text"]
            else:  # directly return the full object
                ret = call_return
        else:
            ret = None
        # --
        assert ret is not None, f"Calling failed for {_call_target_type}"
        if self.print_call_out:
            ss = [f"# == Calling result [ctime={time.ctime()}, interval={time.perf_counter() - time0:.3f}s] =>\n", (ret, self.print_call_out), "\n# =="]
            rprint(ss)
        return ret

    def _call_openai_chat(self, messages, **kwargs):
        _gpt_kwargs = {"model": self.call_target.split(":", 1)[1]}
        _gpt_kwargs.update(kwargs)
        while True:
            try:
                ret = OpenaiHelper().call_chat(messages, stat=self.call_stat, **_gpt_kwargs)
                return ret
            except Exception as e:  # simply catch everything!
                rprint(f"Get error when calling gpt: {e}", style="white on red")
                if type(e).__name__ in ["RateLimitError"]:
                    time.sleep(10)
                elif type(e).__name__ == "BadRequestError":
                    error_str = str(e)
                    if "ResponsibleAIPolicyViolation" in error_str or "content_filter" in error_str:
                        return "Thought: Jailbreak or content filter violation detected. Please modify your prompt or stop with N/A."
                    else:
                        rprint(f"BadRequestError: {error_str}", style="white on red")
                    break
                else:
                    break
        return None

    def _call_claude_chat(self, messages, **kwargs):
        _claude_kwargs = {"model": self.call_target.split(":", 1)[1]}
        _claude_kwargs.update(kwargs)
        import botocore
        while True:
            try:
                ret = Boto3Helper().call_chat(messages, stat=self.call_stat, **_claude_kwargs)
                return ret
            except botocore.exceptions.ClientError as e:
                rprint(f"Get error when calling gpt: {e}", style="white on red")
                if e.response['Error']['Code'] == 'LimitExceededException':
                    time.sleep(10)
                else:
                    return f"Error calling Claude: {e}"
        return None

    def _call_azure_claude_chat(self, messages, **kwargs):
        _azure_claude_kwargs = {"model": self.call_target.split(":", 1)[1]}
        _azure_claude_kwargs.update(kwargs)
        while True:
            try:
                ret = AzureClaudeHelper().call_chat(messages, stat=self.call_stat, **_azure_claude_kwargs)
                return ret
            except Exception as e:  # simply catch everything!
                rprint(f"Get error when calling Azure Claude: {e}", style="white on red")
                if type(e).__name__ in ["RateLimitError"]:
                    time.sleep(10)
                elif type(e).__name__ == "BadRequestError":
                    error_str = str(e)
                    if "ResponsibleAIPolicyViolation" in error_str or "content_filter" in error_str:
                        return "Thought: Jailbreak or content filter violation detected. Please modify your prompt or stop with N/A."
                    else:
                        rprint(f"BadRequestError: {error_str}", style="white on red")
                    break
                else:
                    break
        return None


# --
def test_llm():
    llm = LLM(call_target="gpt:gpt-4o-mini")
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    while True:
        p = input("Prompt >> ")
        messages.append({"role": "user", "content": p.strip()})
        r = llm(messages)
        messages.append({"role": "assistant", "content": r})

if __name__ == '__main__':
    test_llm()