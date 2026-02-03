#

# utils for our web-agent

import re
import io
import os
import copy
import requests
import base64
import pdf2image
import base64
import math
import ast

from ..agents.utils import KwargsInitializable, rprint, zwarn, zlog
from .mdconvert import MarkdownConverter
import markdownify

def check_file_size(file_path, max_size=100):
    """
    Checks if a file's size is greater than max_size Megabytes.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        None. Prints the result to the console.
    """
    # Define the size threshold in bytes (512 MB)
    # 1 MB = 1024 * 1024 bytes
    threshold_bytes = max_size * 1024 * 1024

    try:
        # Get the file size in bytes
        file_size_bytes = os.path.getsize(file_path)
        
        rprint(f"Size: {file_size_bytes / (1024*1024):.2f} MB")

        # Compare the file size to the threshold
        if file_size_bytes > threshold_bytes:
            return True
        else:
            return False

    except FileNotFoundError:
        rprint(f"Error: The file '{file_path}' was not found.", stype="white on red")
        return False
    except Exception as e:
        rprint(f"An error occurred: {e}", stype="white on red")
        return False

class MyMarkdownify(markdownify.MarkdownConverter):
    def convert_img(self, el, text, parent_tags):
        return ""  # simply ignore image

    def convert_a(self, el, text, parent_tags):
        if (not text) or (not text.strip()):
            return ""  # empty
        text = text.strip()  # simply strip!
        href = el.get("href")
        if not href:
            href = ""
        if not any(href.startswith(z) for z in ["http", "https"]):
            ret = text  # simply no links
            # ret = ""  # more aggressively remove things! (nope, removing too much...)
        else:
            ret = f"[{text}]({href})"
        return ret

    @staticmethod
    def md_convert(html: str):
        html_md = MyMarkdownify().convert(html)
        valid_lines = []
        for line in html_md.split("\n"):
            line = line.rstrip()
            if not line: continue
            valid_lines.append(line)
        ret = "\n".join(valid_lines)
        return ret

# --
# web state
class FileState:
    def __init__(self, **kwargs):
        # current file
        self.current_file_name = None
        self.multimodal = False # whether to get the multimodal content of this state.
        

        # 

        self.loaded_files = {} # keys: file names, values: True/False, whether the file is loaded.
        self.file_meta_data = {} # A string indicating number of pages, tokens each page.
        self.current_page_id_list = []
        
        # 
        
        self.textual_content = ""
        self.visual_content = []
        self.image_suffix = []
        
        # step info
        self.curr_step = 0  # step to the root
        self.total_actual_step = 0  # [no-rev] total actual steps including reverting (can serve as ID)
        self.num_revert_state = 0  # [no-rev] number of state reversion
        # (last) action information
        self.action_string = ""
        self.action = None
        self.error_message = ""
        self.observation = ""
        # --
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.__dict__), f"Attribute not found for {k} <- {v}"
        self.__dict__.update(**kwargs)

    def to_dict(self):
        return self.__dict__.copy()

    def copy(self):
        return FileState(**self.to_dict())

    def __repr__(self):
        return f"FileState({self.__dict__})"

# an opened web browser
class FileEnv(KwargsInitializable):
    def __init__(self, starting=True, starting_file_path_dict=None, **kwargs):
        # self.file_path_dict = starting_file_path_dict if starting_file_path_dict else {}  # store these in the state instead
        self.md_converter = MarkdownConverter()
        self.file_text_by_page = {}
        self.file_screenshot_by_page = {}
        self.file_token_num_by_page = {}
        self.file_image_suffix_by_page = {}

        # maximum number of tokens that can be processed by the File Agent LLM
        self.max_file_read_tokens = 2000
        self.max_file_screenshots = 2 
        # these variables will be overrwitten by that in kwargs.

        super().__init__(**kwargs)
        # --
        self.state: FileState = None
        if starting:
            self.start(starting_file_path_dict)  # start at the beginning
        # --

    def read_file_by_page_text(self, file_path: str):
        return self.md_converter.convert(file_path).text_content.split('\x0c') # split by pages

    def find_file_name(self, file_name):
        # this function returns an exact match or a fuzzy match of the LLM-output file_name and what the files the environment actually have in state.loaded_files
        file_path_dict = self.state.loaded_files
        if file_name in file_path_dict:  # directly matching
            return file_name
        elif os.path.basename(file_name) in [os.path.basename(p) for p in file_path_dict]:  # allow name matching
            return [p for p in file_path_dict if os.path.basename(p) == os.path.basename(file_name)][0]
        elif os.path.exists(file_name):
            self.add_files_to_load([file_name])  # add it!
            return file_name
        else:  # file not found!
            raise FileNotFoundError(f"FileNotFoundError for {file_name}.")

    @staticmethod
    def read_file_by_page_screenshot(file_path: str):

        screenshots_b64 = []
        if file_path.endswith(".pdf"):
        
            images = pdf2image.convert_from_path(file_path)
            screenshots_b64 = []

            # Let's use the first page as an example
            for img in images:

                # Save the image to a bytes buffer in PNG format
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                img_bytes = buffer.read()

                # Encode to base64
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                screenshots_b64.append(img_b64)
        if file_path.endswith(".xlsx") or file_path.endswith(".xls") or file_path.endswith(".csv"):
            import subprocess

            input_file = file_path
            
            subprocess.run([
                "soffice", "--headless", "--convert-to", "pdf", "--outdir",
                os.path.dirname(input_file), input_file
            ])

            if input_file.endswith(".xlsx"):
                pdf_file = input_file[:-5] + ".pdf"
            elif input_file.endswith(".xls"):
                pdf_file = input_file[:-4] + ".pdf"
            elif input_file.endswith(".csv"):
                pdf_file = input_file[:-4] + ".pdf"

            images = pdf2image.convert_from_path(pdf_file)
            screenshots_b64 = []

            # Let's use the first page as an example
            for img in images:

                # Save the image to a bytes buffer in PNG format
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                img_bytes = buffer.read()

                # Encode to base64
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                screenshots_b64.append(img_b64)
        


        return screenshots_b64

    def start(self, file_path_dict=None):
        # for file_path in file_path_dict:
        #     self.file_text_by_page[file_path] = self.read_file_by_page_text(file_path=file_path)
        #     self.file_screenshot_by_page[file_path] = FileEnv.read_file_by_page_screenshot(file_path=file_path)
        self.init_state(file_path_dict)

    def stop(self):
        if self.state is not None:
            self.end_state()
            self.state = None

    def __del__(self):
        self.stop()

    # note: return a copy!
    def get_state(self, export_to_dict=True, return_copy=True):
        assert self.state is not None, "Current state is None, should first start it!"
        if export_to_dict:
            ret = self.state.to_dict()
        elif return_copy:
            ret = self.state.copy()
        else:
            ret = self.state
        return ret

    # --
    # helpers

    def parse_action_string(self, action_string, state):
        patterns = {
            "load_file": r'load_file\((.*)\)',
            "read_text": r'read_text\((.*)\)',
            "read_screenshot": r'read_screenshot\((.*)\)',
            "search": r'search\((.*)\)',
            "stop": r"stop(.*)",
            "nop": r"nop(.*)",
        }
        action = {"action_name": "", "target_file": None, "page_id_list": None, "key_word_list": None}  # assuming these fields
        if action_string:
            for key, pat in patterns.items():
                m = re.match(pat, action_string, flags=(re.IGNORECASE|re.DOTALL))  # ignore case and allow \n
                if m:
                    action["action_name"] = key
                    if key in ["read_text", "read_screenshot"]:
                        args_str = m.group(1)  # target ID
                        m_file = re.search(r'file_name\s*=\s*(".*?"|\'.*?\'|\[.*?\]|\d+)', args_str)
                        m_page = re.search(r'page_id_list\s*=\s*(".*?"|\'.*?\'|\[.*?\]|\d+)', args_str)
                        if m_file:
                            file_name = m_file.group(1)
                        else:
                            file_name = None
                        if m_page:
                            page_id_list = m_page.group(1)
                        else:
                            page_id_list = None
                        
                        # If not named, try positional
                        if file_name is None or page_id_list is None:
                            # Split by comma not inside brackets or quotes
                            # This is a simple split, not perfect for all edge cases
                            parts = re.split(r',(?![^\[\]]*\])', args_str)
                            if len(parts) >= 2:
                                if file_name is None:
                                    file_name = parts[0]
                                if page_id_list is None:
                                    page_id_list = parts[1]

                        # Clean up quotes if needed
                        if file_name:
                            file_name = file_name.strip('\'"')
                        if page_id_list:
                            page_id_list = page_id_list.strip()

                        # 
                        if file_name is None or page_id_list is None:
                            zwarn(f"Failed to parse action string: {action_string}")
                            return {"action_name": None}
                        
                        action["target_file"] = file_name.strip('"').strip("'")
                        action["page_id_list"] = page_id_list
                    elif key == "search":
                        # search("filename.pdf", ["xxx", "yyy"])
                        # search("filename.pdf", ['xxx', 'yyy'])
                        # search("filename.pdf", ["xxx", 'yyy'])
                        # search("filename.pdf", "xxx")
                        # search(file_name.pdf, "xxx")
                        # search(file_name="filename.pdf", ["xxx", 'yyy'])
                        # search(file_name="filename.pdf", key_word_list=["xxx", 'yyy'])
                        s = m.group(1) 
                        
                        filename_match = re.search(
                            r'(?:file_name\s*=\s*)?'
                            r'(?:["\']([\w\-.]+\.pdf)["\']|([\w\-.]+\.pdf))', s)
                        filename = None
                        if filename_match:
                            filename = filename_match.group(1) or filename_match.group(2)

                        # Match keywords: list or string, positional or keyword argument
                        keyword_match = re.search(
                            r'(?:key_word_list\s*=\s*|,\s*)('
                            r'\[[^\]]+\]|'      # a list: [ ... ]
                            r'["\'][^"\']+["\']' # or a single quoted string
                            r')', s)
                        keywords = None
                        if keyword_match:
                            kw_str = keyword_match.group(1)
                            try:
                                keywords = ast.literal_eval(kw_str)
                                if isinstance(keywords, str):
                                    keywords = [keywords]
                            except Exception:
                                keywords = [kw_str.strip('"\'')]
                        
                        action["target_file"] = filename
                        if isinstance(keywords, list):
                            action["key_word_list"] = keywords
                        else:
                            action["key_word_list"] = "###Error: the generated key_word_list is not valid. Please retry!"

                    else:
                        action["target_file"] = m.group(1).strip().strip('"').strip("'")

                    if key in ["stop", "nop"]:
                        action["action_value"] = m.groups()[-1].strip()  # target value
                    break
        return action


    def action(self, action):
        file_name = ""
        page_id_list = []
        multimodal = False
        loaded_files = copy.deepcopy(self.state.loaded_files)
        file_meta_data = copy.deepcopy(self.state.file_meta_data)
        visual_content = None
        image_suffix = None
        error_message = None
        textual_content = ""
        observation = None

        if action["action_name"] == "load_file":
            file_name = self.find_file_name(action["target_file"])

            # if the file exceeds 512MB
            max_file_size = 100 if file_name.endswith(".pdf") else 20
            if check_file_size(file_name, max_file_size):
                return False, {"current_file_name": file_name, 
                                "current_page_id_list": [], 
                                "loaded_files": [], 
                                "multimodal": multimodal, 
                                "file_meta_data": [], 
                                "textual_content": "The file is too big to be loaded. Please skip this file and return None.", 
                                "visual_content": "", 
                                "image_suffix": "", 
                                "error_message": "The file is too big to be loaded. Please skip this file and return None.", 
                                "observation": ""}

            

            if file_name.endswith(".pdf"):
                text_pages = self.md_converter.convert(file_name).text_content.split('\x0c') # split by pages
                text_screenshots = FileEnv.read_file_by_page_screenshot(file_name)
                _page_token_num = [math.ceil(len(text_pages[i].encode())/4) for i in range(len(text_pages))]
                _info = ", ".join([f"Sheet {i}: {  _page_token_num[i]  } "  for i in range(len(text_pages))])
                file_meta_data[file_name] = f"Number of pages of {file_name}: {len(text_pages)}. Number of tokens of each page: {_info}"
                observation = f"load_file({file_name})  # number of pages is {len(text_pages)}"
                image_suffix = ['png' for _ in text_screenshots]
            elif file_name.endswith(".xlsx") or file_name.endswith(".xls") or file_name.endswith(".csv"):
                text_pages = self.md_converter.convert(file_name).text_content.split('\x0c') # split by sheets
                text_screenshots = FileEnv.read_file_by_page_screenshot(file_name)
                _page_token_num = [math.ceil(len(text_pages[i].encode())/4) for i in range(len(text_pages))]
                _info = ", ".join([f"Sheet {i}: {  _page_token_num[i]  } "  for i in range(len(text_pages))])
                file_meta_data[file_name] = f"Number of sheets of {file_name}: {len(text_pages)}. Number of tokens of each page: {_info}. Number of screenshots of the excel file: {len(text_screenshots)}"
                observation = f"load_file({file_name})  # number of sheets is {len(text_pages)}"
                image_suffix = ['png' for _ in text_screenshots]
            elif any(file_name.endswith(img_suffix) for img_suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']):
                text_pages = [""]
                _page_token_num = [0]
                with open(file_name, 'rb') as f:
                    img_bytes = f.read()
                # Base64-encode the bytes and decode to UTF-8 string
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                text_screenshots = [img_b64]
                image_suffix = [file_name.split('.')[-1]]
                file_meta_data[file_name] = "This is an image."
                observation = f"load_file({file_name})  # load an image"
            else:
                # first, try to use markdown converter to load the file
                # breakpoint()
                content = self.md_converter.convert(file_name)
                if any(file_name.endswith(img_suffix) for img_suffix in ['.htm', '.html']):
                    content = MyMarkdownify().md_convert(content.text_content)
                else:
                    content = content.text_content
                    
                if '\x0c' in content:
                    text_pages = content.split('\x0c') # split by pages
                else:
                    def split_text_to_pages(text, max_tokens_per_page):
                        """
                        Split the text into pages where each page has approximately max_tokens_per_page tokens.

                        :param text: The input text to be split.
                        :param max_tokens_per_page: The maximum number of tokens per page.
                        :return: A list of text pages.
                        """
                        # Initialize variables
                        pages = []
                        current_page = []
                        current_tokens = 0

                        # Split the text into words
                        words = text.split()

                        for word in words:
                            # Estimate the number of tokens for the current word
                            word_tokens = math.ceil(len(word.encode()) / 4)

                            # Check if adding this word would exceed the max tokens per page
                            if current_tokens + word_tokens > max_tokens_per_page:
                                # If so, finalize the current page and start a new one
                                pages.append(' '.join(current_page))
                                current_page = [word]
                                current_tokens = word_tokens
                            else:
                                # Otherwise, add the word to the current page
                                current_page.append(word)
                                current_tokens += word_tokens

                        # Add the last page if it contains any words
                        if current_page:
                            pages.append(' '.join(current_page))

                        return pages
                    
                    text_pages = split_text_to_pages(content, self.max_file_read_tokens)
                # text_screenshots = FileEnv.read_file_by_page_screenshot(file_name)
                text_screenshots = []
                _page_token_num = [math.ceil(len(text_pages[i].encode())/4) for i in range(len(text_pages))]
                _info = ", ".join([f"Sheet {i}: {  _page_token_num[i]  } "  for i in range(len(text_pages))])
                file_meta_data[file_name] = f"Number of pages of {file_name}: {len(text_pages)}. Number of tokens of each page: {_info}. Number of screenshots of the excel file: {len(text_screenshots)}"
                observation = f"load_file({file_name})  # number of sheets is {len(text_pages)}"


            loaded_files[file_name]= True

            # save the info to the file env
            self.file_text_by_page[file_name] = text_pages
            self.file_token_num_by_page[file_name] = _page_token_num
            self.file_screenshot_by_page[file_name] = text_screenshots
            self.file_image_suffix_by_page[file_name] = image_suffix

            page_id_list = []

            textual_content = "The file has just loaded. Please call read_text() or read_screenshot()."

        elif action["action_name"] == "read_text":
            file_name = self.find_file_name(action["target_file"])
            visual_content = None
            page_id_list = eval(action["page_id_list"])
            # Check if the total number of tokens exceed max_file_read_tokens
            total_token_num = sum([self.file_token_num_by_page[file_name][i] for i in page_id_list])
            truncated_page_id_list = []
            remaining_page_id_list = []
            if total_token_num > self.max_file_read_tokens:
                for j in range(len(page_id_list)-1, 0, -1):
                    if sum([self.file_token_num_by_page[file_name][i] for i in page_id_list[:j]]) <= self.max_file_read_tokens:
                        truncated_page_id_list = page_id_list[:j]
                        remaining_page_id_list = page_id_list[j:]
                        break
                # textual_content = "\n\n".join([f"Page {i}\n" + self.file_text_by_page[file_name][i] for i in page_id_list])
                error_message = f"The pages you selected ({page_id_list}) exceed the maximum token limit {self.max_file_read_tokens}. They have been truncated to {truncated_page_id_list}.  {remaining_page_id_list} has not been reviewed."
                page_id_list = truncated_page_id_list
            # else:
            textual_content = "\n\n".join([f"Page {i}\n" + self.file_text_by_page[file_name][i] for i in page_id_list])
            multimodal = False
            observation = f"read_text({file_name}, {page_id_list})  # Read {len(page_id_list)} pages"
        elif action["action_name"] == "read_screenshot":
            
            file_name = self.find_file_name(action["target_file"])
            page_id_list = eval(action["page_id_list"])
            textual_content = "\n\n".join([f"Page {i}\n" + self.file_text_by_page[file_name][i] for i in page_id_list])
            
            # make sure the number of screenshots and total number of text tokens both do not exceed the maximum constraint.
            truncated_page_id_list = copy.deepcopy(page_id_list)
            remaining_page_id_list = []
            if len(page_id_list) > self.max_file_screenshots:
                truncated_page_id_list = truncated_page_id_list[:self.max_file_screenshots]
                remaining_page_id_list = sorted(list(set(page_id_list) - set(truncated_page_id_list)))
            
            # check if text tokens satisfy the contraint:
            if sum([self.file_token_num_by_page[file_name][i] for i in truncated_page_id_list]) > self.max_file_read_tokens:
                for j in range(len(truncated_page_id_list)-1, 0, -1):
                    if sum([self.file_token_num_by_page[file_name][i] for i in truncated_page_id_list[:j]]) <= self.max_file_read_tokens:
                        
                        truncated_page_id_list = truncated_page_id_list[:j]
                        remaining_page_id_list = sorted(list(set(page_id_list) - set(truncated_page_id_list)))
                        break
            
            
            if len(remaining_page_id_list) > 0:
                error_message = f"The pages you selected ({page_id_list}) exceed the maximum token limit {self.max_file_read_tokens} or the maximum screenshot limit {self.max_file_screenshots}. They have been truncated to {truncated_page_id_list}. {remaining_page_id_list} has not been reviewed."
                page_id_list = truncated_page_id_list
            
            textual_content = "\n\n".join([f"Page {i}\n" + self.file_text_by_page[file_name][i] for i in page_id_list])

            visual_content = [self.file_screenshot_by_page[file_name][i] for i in page_id_list]
            image_suffix = [self.file_image_suffix_by_page[file_name][i] for i in page_id_list]
            multimodal = True
            observation = f"read_screenshot({file_name}, {page_id_list})  # Read {len(page_id_list)} pages"
        elif action["action_name"] == "search":
            if "###Error" in action["key_word_list"]:
                error_message = action["key_word_list"]
            else:
                # perform searching 
                file_name = self.find_file_name(action["target_file"])
                key_word_list = action["key_word_list"]

                def find_keyword_pages(file_name, key_word_list):
                    """
                    file_text_by_page: dict, e.g. {'filename.pdf': [page1_text, page2_text, ...]}
                    file_name: str, the filename key
                    key_word_list: list of str, keywords to search for
                    page_base: 0 for 0-based page numbers, 1 for 1-based
                    Returns: dict, {keyword: [page_numbers]}
                    """
                    result = {}
                    pages = self.file_text_by_page[file_name]
                    for keyword in key_word_list:
                        result[keyword] = [
                            i for i, page_text in enumerate(pages)
                            if keyword in page_text
                        ]
                    return result

                search_result = find_keyword_pages(file_name, key_word_list)
                observation = f"The result of search({file_name}, {key_word_list}). The keys of the result dict are the keywords, and the values are the corresponding page indices that contains the keyword: {search_result}"

        elif action["action_name"] == "stop":
            pass
        
        # self.state.current_file_name = file_name
        # self.state.current_page_id_list = page_id_list
        if error_message:
            observation = f"{observation} (**Warning**: {error_message})"

        return True, {"current_file_name": file_name, "current_page_id_list": page_id_list, "loaded_files": loaded_files, "multimodal": multimodal, "file_meta_data": file_meta_data, "textual_content": textual_content, "visual_content": visual_content, "image_suffix": image_suffix, "error_message": error_message, "observation": observation}
        
    # --
    # other helpers

    # --
    # main step

    def init_state(self, file_path_dict: dict):
        self.state = FileState()  # set the new state!
        if file_path_dict:
            self.add_files_to_load(file_path_dict)

    def end_state(self):
        del self.file_text_by_page
        del self.file_screenshot_by_page
        import gc
        gc.collect()

    def add_files_to_load(self, files):
        self.state.loaded_files.update({file: False for file in files})

    def step_state(self, action_string: str):
        state = self.state
        action_string = action_string.strip()
        # --
        # parse action
        action = self.parse_action_string(action_string, state)
        
        zlog(f"[CallFile:{state.curr_step}:{state.total_actual_step}] ACTION={action} ACTION_STR={action_string}", timed=True)
        # --
        # execution
        state.curr_step += 1
        state.total_actual_step += 1
        state.update(action=action, action_string=action_string, error_message="")  # first update some of the things
        if not action["action_name"]:  # UNK action
            state.error_message = f"The action you previously choose is not well-formatted: {action_string}. Please double-check if you have selected the correct element or used correct action format."
            ret = state.error_message
        elif action["action_name"] in ["stop", "nop"]:  # ok, nothing to do
            ret = f"File agent step: {action_string}"
        else:
            # actually perform action
            action_succeed, results  = self.action(action)
            if not action_succeed:  # no succeed
                state.error_message = f"The action you have chosen cannot be executed: {action_string}. Please double-check if you have selected the correct element or used correct action format. {results['error_message']}"
                ret = state.error_message
            else:  # get new states
                # results = self._get_current_file_state(state)
                state.update(**results)  # update it!
                ret = f"File agent step: {results.get('observation', action_string)}"
        return ret
        # --
