#

# utils for our web-agent

import re
import os
import subprocess
import signal
import time
import requests
import base64
import markdownify
from ..agents.utils import KwargsInitializable, rprint, zwarn, zlog

# --
# web state
class WebState:
    def __init__(self, **kwargs):
        # not-changed
        self.browser_id = ""
        self.page_id = ""
        self.target_url = ""
        # from tree-results
        self.get_accessibility_tree_succeed = False
        self.current_accessibility_tree = ""
        self.step_url = ""
        self.html_md = ""
        self.snapshot = ""
        self.boxed_screenshot = ""  # always store the screenshot here
        self.downloaded_file_path = []
        self.current_has_cookie_popup = False
        self.expanded_part = None
        # step info
        self.curr_step = 0  # step to the root
        self.curr_screenshot_mode = False  # whether we are using screenshot or not?
        self.total_actual_step = 0  # [no-rev] total actual steps including reverting (can serve as ID)
        self.num_revert_state = 0  # [no-rev] number of state reversion
        # (last) action information
        self.action_string = ""
        self.action = None
        self.error_message = ""
        # --
        self.update(**kwargs)

    def get_id(self):  # use these as ID
        return (self.browser_id, self.page_id, self.total_actual_step)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.__dict__), f"Attribute not found for {k} <- {v}"
        self.__dict__.update(**kwargs)

    def to_dict(self):
        return self.__dict__.copy()

    def copy(self):
        return WebState(**self.to_dict())

    def __repr__(self):
        return f"WebState({self.__dict__})"

# --
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

# an opened web browser
class WebEnv(KwargsInitializable):
    def __init__(self, starting=True, starting_target_url=None, **kwargs):
        self.web_ip = os.getenv("WEB_IP", "localhost:3000")  # allow set by ENV
        self.web_command = ""  # if running a local one
        self.web_timeout = 600  # set a timeout!
        # self.use_screenshot = False  # add screenshot? -> for simplicity, always store it!
        self.screenshot_boxed = True  # use boxed or nonboxed
        # self.target_url = "https://www.google.com/?hl=en"  # by default
        self.target_url = "https://www.bing.com/"  # by default
        # self.target_url = "https://duckduckgo.com/"  # by default
        # --
        super().__init__(**kwargs)
        # --
        self.state: WebState = None
        self.popen = None  # popen obj for subprocess running
        if starting:
            self.start(starting_target_url)  # start at the beginning
        # --

    def start(self, target_url=None):
        self.stop()  # stop first
        # --
        # optionally start one
        if self.web_command:
            self.popen = subprocess.Popen(self.web_command, shell=True, preexec_fn=os.setsid)  # make a new one
            time.sleep(15)  # wait for some time
            rprint(f"Web-Utils-Start {self.popen}")
        # --
        target_url = target_url if target_url is not None else self.target_url  # otherwise use default
        ### hard code: replace google to bing
        if 'www.google.com' in target_url:
            if not 'www.google.com/maps' in target_url:
                target_url = target_url.replace('www.google.com', 'www.bing.com')
        self.init_state(target_url)

    def stop(self):
        if self.state is not None:
            self.end_state()
            self.state = None
        if self.popen is not None:
            try:
                # Try to terminate gracefully first
                self.popen.terminate()
                time.sleep(2)  # wait for graceful shutdown
                if self.popen.poll() is None:
                    # If still running, try to kill the process group
                    try:
                        os.killpg(self.popen.pid, signal.SIGKILL)  # kill the PG
                    except (PermissionError, ProcessLookupError):
                        # If permission denied or process not found, just kill the main process
                        self.popen.kill()
                time.sleep(1)  # slightly wait
            except (PermissionError, ProcessLookupError):
                # Handle cases where we can't kill the process
                rprint(f"Warning: Could not kill process {self.popen.pid}, it may still be running")
            rprint(f"Web-Utils-Kill {self.popen} with {self.popen.poll()}")
            self.popen = None

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

    def get_target_url(self):
        return self.target_url

    # --
    # helpers

    def get_browser(self, storage_state, geo_location):
        url = f"http://{self.web_ip}/getBrowser"
        data = {"storageState": storage_state, "geoLocation": geo_location}
        response = requests.post(url, json=data, timeout=self.web_timeout)
        if response.status_code == 200:
            zlog(f"==> Get browser {response.json()}")
            return response.json()["browserId"]
        else:
            raise requests.RequestException(f"Getting browser failed: {response}")

    def close_browser(self, browser_id):
        url = f"http://{self.web_ip}/closeBrowser"
        data = {"browserId": browser_id}
        zlog(f"==> Closing browser {browser_id}")
        try:  # put try here
            response = requests.post(url, json=data, timeout=self.web_timeout)
            if response.status_code == 200:
                return None
            else:
                zwarn(f"Bad response when closing browser: {response}")
        except requests.RequestException as e:
            zwarn(f"Request Error: {e}")
        return None

    def open_page(self, browser_id, target_url):
        url = f"http://{self.web_ip}/openPage"
        data = {"browserId": browser_id, "url": target_url}
        response = requests.post(url, json=data, timeout=self.web_timeout)
        if response.status_code == 200:
            return response.json()["pageId"]
        else:
            raise requests.RequestException(f"Open page Request failed: {response}")

    def goto_url(self, browser_id, page_id, target_url):
        url = f"http://{self.web_ip}/gotoUrl"
        data = {"browserId": browser_id, "pageId": page_id, "targetUrl": target_url}
        response = requests.post(url, json=data, timeout=self.web_timeout)
        if response.status_code == 200:
            return True
        else:
            raise requests.RequestException(f"GOTO page Request failed: {response}")

    def process_html(self, html: str):
        if not html.strip():
            return html  # empty
        return MyMarkdownify.md_convert(html)

    def process_axtree(self, res_json):
        # --
        def _parse_tree_str(_s):
            if "[2]" in _s:
                _lines = _s.split("[2]", 1)[1].split("\n")
                _lines = [z for z in _lines if z.strip().startswith("[")]
                _lines = [" ".join(z.split()[1:]) for z in _lines]
                return _lines
            else:
                return []
        # --
        def _process_tree_str(_s):
            _s = _s.strip()
            if _s.startswith("Tab 0 (current):"):  # todo(+N): sometimes this line can be strange, simply remove it!
                _s = _s.split("\n", 1)[-1].strip()
            return _s
        # --
        html_md = self.process_html(res_json.get("html", ""))
        AccessibilityTree = _process_tree_str(res_json.get("yaml", ""))
        curr_url = res_json.get("url", "")
        snapshot = res_json.get("snapshot", "")
        fulltree = _process_tree_str(res_json.get("fulltree", ""))
        screenshot = res_json.get("boxed_screenshot", "") if self.screenshot_boxed else res_json.get("nonboxed_screenshot", "")
        downloaded_file_path = res_json.get("downloaded_file_path", [])
        all_at, all_ft = _parse_tree_str(AccessibilityTree), _parse_tree_str(fulltree)
        # all_ft_map = {v: i for i, v in enumerate(all_ft)}
        all_ft_map = {}
        for ii, vv in enumerate(all_ft):
            if vv not in all_ft_map:  # no overwritten to get the minumum one
                all_ft_map[vv] = ii
        _hit_at_idxes = [all_ft_map[z] for z in all_at if z in all_ft_map]
        if _hit_at_idxes:
            _last_hit_idx = max(_hit_at_idxes)
            _remaining = len(all_ft) - (_last_hit_idx + 1)
            if _remaining >= len(_hit_at_idxes) * 0.5:  # note: a simple heuristic
                AccessibilityTree = AccessibilityTree.strip() + "\n(* Scroll down to see more items)"
        # --
        ret = {"current_accessibility_tree": AccessibilityTree, "step_url": curr_url, "html_md": html_md, "snapshot": snapshot, "boxed_screenshot": screenshot, "downloaded_file_path": downloaded_file_path}
        return ret

    def get_accessibility_tree(self, browser_id, page_id, current_round):
        url = f"http://{self.web_ip}/getAccessibilityTree"
        data = {
            "browserId": browser_id,
            "pageId": page_id,
            "currentRound": current_round,
        }
        default_axtree = ""  # default empty
        default_res = {"current_accessibility_tree": default_axtree, "step_url": "", "html_md": "", "snapshot": "", "boxed_screenshot": "", "downloaded_file_path": []}
        try:
            response = requests.post(url, json=data, timeout=self.web_timeout)
            if response.status_code == 200:
                res_json = response.json()
                res_dict = self.process_axtree(res_json)
                return True, res_dict
            else:
                zwarn(f"Get accessibility tree Request failed with status code: {response.status_code}")
                return False, default_res
        except requests.RequestException as e:
            zwarn(f"Request failed: {e}")
            return False, default_res

    def action(self, browser_id, page_id, action):
        url = f"http://{self.web_ip}/performAction"
        data = {
            "browserId": browser_id,
            "pageId": page_id,
            "actionName": action["action_name"],
            "targetId": action["target_id"],
            "targetElementType": action["target_element_type"],
            "targetElementName": action["target_element_name"],
            "actionValue": action["action_value"],
            "needEnter": action["need_enter"],
        }
        try:
            response = requests.post(url, json=data, timeout=self.web_timeout)
            if response.status_code == 200:
                return True
            else:
                zwarn(f"Request failed with status code: {response.status_code} {response.text}")
                return False
        except requests.RequestException as e:
            zwarn(f"Request failed: {e}")
            return False

    # --
    # other helpers

    def is_annoying(self, current_accessbility_tree):
        if "See results closer to you?" in current_accessbility_tree and len(current_accessbility_tree.split("\n")) <= 10:
            return True
        return False

    def parse_action_string(self, action_string: str, state):
        patterns = {"click": r"click\s+\[?(\d+)\]?", "type": r"type\s+\[?(\d+)\]?\s+\{?(.+)\}?", "scroll": r"scroll\s+(down|up)", "wait": "wait", "goback": "goback", "restart": "restart", "stop": r"stop(.*)", "goto": r"goto(.*)", "save": r"save(.*)", "screenshot": r"screenshot(.*)", "nop": r"nop(.*)"}
        action = {"action_name": "", "target_id": None, "action_value": None, "need_enter": None, "target_element_type": None, "target_element_name": None}  # assuming these fields
        if action_string:
            for key, pat in patterns.items():
                m = re.match(pat, action_string, flags=(re.IGNORECASE|re.DOTALL))  # ignore case and allow \n
                if m:
                    action["action_name"] = key
                    if key in ["click", "type"]:
                        action["target_id"] = m.groups()[0]  # target ID
                    if key in ["type", "scroll", "stop", "goto", "save", "screenshot"]:
                        action["action_value"] = m.groups()[-1].strip()  # target value
                        if key == "type":  # quick fix
                            action["action_value"] = action["action_value"].rstrip("}]").rstrip().strip("\"'").strip()
                    # if key == "restart":
                    #     action["action_value"] = state.target_url  # restart
                    break
        return action

    @staticmethod
    def find_target_element_info(current_accessibility_tree, target_id, action_name):
        if target_id is None:
            return None, None, None
        if action_name == "type":
            tree_to_check = current_accessibility_tree.split("\n")[int(target_id) - 1:]
            for i, line in enumerate(tree_to_check):
                if f"[{target_id}]" in line and ("combobox" in line or "box" not in line):
                    num_tabs = len(line) - len(line.lstrip("\t"))
                    for j in range(i + 1, len(tree_to_check)):
                        curr_num_tabs = len(tree_to_check[j]) - len(tree_to_check[j].lstrip("\t"))
                        if curr_num_tabs <= num_tabs:
                            break
                        if "textbox" in tree_to_check[j] or "searchbox" in tree_to_check[j]:
                            target_element_id = tree_to_check[j].split("]")[0].strip()[1:]
                            # print("CATCHED ONE MISSED TYPE ACTION, changing the type action to", target_element_id)
                            target_id = target_element_id
        target_pattern = r"\[" + re.escape(target_id) + r"\] ([a-z]+) '(.*)'"
        matches = re.finditer(target_pattern, current_accessibility_tree, re.IGNORECASE)
        for match in matches:
            target_element_type, target_element_name = match.groups()
            return target_id, target_element_type, target_element_name
        return target_id, None, None

    @staticmethod
    def get_skip_action(current_accessbility_tree):
        # action_name, target_id, action_value, need_enter = extract_info_from_action("click [5]")
        action_name, target_id, action_value, need_enter = "click", "5", "", None
        target_id, target_element_type, target_element_name = WebEnv.find_target_element_info(current_accessbility_tree, target_id, action_name)
        return {
            "action_name": action_name,
            "target_id": target_id,
            "action_value": action_value,
            "need_enter": need_enter,
            "target_element_type": target_element_type,
            "target_element_name": target_element_name,
        }

    @staticmethod
    def check_if_menu_is_expanded(accessibility_tree, snapshot):
        node_to_expand = {}
        lines = accessibility_tree.split("\n")
        for i, line in enumerate(lines):
            if 'hasPopup: menu' in line and 'expanded: true' in line:
                num_tabs = len(line) - len(line.lstrip("\t"))
                next_tabs = len(lines[i + 1]) - len(lines[i + 1].lstrip("\t"))
                if next_tabs <= num_tabs:
                    # In this case, the menu should be expanded but is not present in the tree
                    target_pattern = r"\[(\d+)\] ([a-z]+) '(.*)'"
                    matches = re.finditer(target_pattern, line, re.IGNORECASE)
                    target_id = None
                    target_element_type = None
                    target_element_name = None
                    for match in matches:
                        target_id, target_element_type, target_element_name = match.groups()
                        break
                    if target_element_type is not None:
                        # locate the menu items from the snapshot instead
                        children = WebEnv.find_node_with_children(snapshot, target_element_type, target_element_name)
                        if children is not None:
                            node_to_expand[i] = (num_tabs + 1, children, target_id, target_element_type, target_element_name)
        new_lines = []
        curr = 1
        if len(node_to_expand) == 0:
            return accessibility_tree, None
        expanded_part = {}
        # add the menu items to the correct location in the tree
        for i, line in enumerate(lines):
            if not line.strip().startswith('['):
                new_lines.append(line)
                continue
            num_tabs = len(line) - len(line.lstrip("\t"))
            content = line.split('] ')[1]
            new_lines.append('\t' * num_tabs + f"[{curr}] {content}")
            curr += 1
            if i in node_to_expand:
                for child in node_to_expand[i][1]:
                    child_content = f"{child.get('role', '')} '{child.get('name', '')}' " + ' '.join([f"{k}: {v}" for k, v in child.items() if k not in ['role', 'name']])
                    tabs = '\t' * node_to_expand[i][0]
                    new_lines.append(f"{tabs}[{curr}] {child_content}")
                    expanded_part[curr] = (node_to_expand[i][2], node_to_expand[i][3], node_to_expand[i][4])
                    curr += 1
        return '\n'.join(new_lines), expanded_part

    @staticmethod
    def find_node_with_children(node, target_role, target_name):
        # Check if the current node matches the target role and name
        if node.get('role') == target_role and node.get('name') == target_name:
            return node.get('children', None)
        # If the node has children, recursively search through them
        children = node.get('children', [])
        for child in children:
            result = WebEnv.find_node_with_children(child, target_role, target_name)
            if result is not None:
                return result
        # If no matching node is found, return None
        return None

    # --
    # main step

    def init_state(self, target_url: str):
        browser_id = self.get_browser(None, None)
        page_id = self.open_page(browser_id, target_url)
        curr_step = 0
        state = WebState(browser_id=browser_id, page_id=page_id, target_url=target_url, curr_step=curr_step, total_actual_step=curr_step)  # start from 0
        results = self._get_accessibility_tree_results(state)
        state.update(**results)  # update it!
        # --
        self.state = state  # set the new state!
        # --

    def end_state(self):
        state = self.state
        self.close_browser(state.browser_id)

    def reset_to_state(self, target_state):
        state = self.state
        if isinstance(target_state, dict):
            target_state = WebState.create_from_dict(target_state)
        # assert state.browser_id == target_state.browser_id and state.page_id == target_state.page_id, "Mismatched basic IDs"
        if state.get_id() != target_state.get_id():  # need to revert to another URL
            self.goto_url(target_state.browser_id, target_state.page_id, target_state.step_url)
            state.update(browser_id=target_state.browser_id, page_id=target_state.page_id)
            results = self._get_accessibility_tree_results(state)
            state.update(**results)  # update it!
            # --
            # revert other state info
            state.update(curr_step=target_state.curr_step, action_string=target_state.action_string, action=target_state.action, error_message=target_state.error_message)  # no change of total_step!
            state.num_revert_state += 1
            # --
            zlog(f"Reset state with URL={target_state.step_url}")
            return True
        else:
            assert state.to_dict() == target_state.to_dict(), "Mismatched state!"
            zlog("No need for state resetting!")
            return False
        # --

    def _get_accessibility_tree_results(self, state):
        get_accessibility_tree_succeed, curr_res = self.get_accessibility_tree(state.browser_id, state.page_id, state.curr_step)
        current_accessibility_tree = curr_res.get("current_accessibility_tree", "")
        if not get_accessibility_tree_succeed:
            zwarn("Failed to get current_accessibility_tree!!")
        if self.is_annoying(current_accessibility_tree):
            skip_this_action = self.get_skip_action(current_accessibility_tree)
            self.action(state.browser_id, state.page_id, skip_this_action)
            get_accessibility_tree_succeed, curr_res = self.get_accessibility_tree(state.browser_id, state.page_id, state.curr_step)
        # try to close cookie popup
        if "Cookie banner" in current_accessibility_tree:
            current_has_cookie_popup = True  # note: only mark here!
        else:
            current_has_cookie_popup = False
        current_accessibility_tree, expanded_part = self.check_if_menu_is_expanded(current_accessibility_tree, curr_res["snapshot"])
        # --
        # if (not self.use_screenshot) and ("boxed_screenshot" in curr_res):  # note: no storing of snapshot since it is too much
        #     del curr_res["boxed_screenshot"]  # for simplicity, always store it
        # --
        # more checking on axtree
        if not current_accessibility_tree or ("[2]" not in current_accessibility_tree):  # at least we should have some elements!
            curr_res["current_accessibility_tree"] = current_accessibility_tree + "\n**Warning**: The accessibility tree is currently unavailable. Please try some alternative actions. If the issue persists after multiple attempts, consider goback or restart."
        # --
        curr_res.update(get_accessibility_tree_succeed=get_accessibility_tree_succeed, current_has_cookie_popup=current_has_cookie_popup, expanded_part=expanded_part)
        return curr_res

    def step_state(self, action_string: str):
        state = self.state
        # --
        need_enter = True
        if "[NOENTER]" in action_string:
            need_enter = False
            action_string = action_string.replace("[NOENTER]", "")  # note: ugly quick fix ...
        # --
        action_string = action_string.strip()
        # parse action
        action = self.parse_action_string(action_string, state)
        if action["action_name"]:
            if action["action_name"] in ["click", "type"]:  # need more handling
                target_id, target_element_type, target_element_name = self.find_target_element_info(state.current_accessibility_tree, action["target_id"], action["action_name"])
                if state.expanded_part and int(target_id) in state.expanded_part:
                    expand_target_id, expand_target_type, expand_target_name = state.expanded_part[int(target_id)]
                    action.update({"action_name": "select", "target_id": expand_target_id, "action_value": target_element_name, "target_element_type": expand_target_type, "target_element_name": expand_target_name})
                else:
                    action.update({"target_id": target_id, "target_element_type": target_element_type, "target_element_name": target_element_name})
            if action["action_name"] == "type":
                action["need_enter"] = need_enter
        zlog(f"[CallWeb:{state.curr_step}:{state.total_actual_step}] ACTION={action} ACTION_STR={action_string}", timed=True)
        # --
        # execution
        state.curr_step += 1
        state.total_actual_step += 1
        state.update(action=action, action_string=action_string, error_message="")  # first update some of the things
        if not action["action_name"]:  # UNK action
            state.error_message = f"The action you previously choose is not well-formatted: {action_string}. Please double-check if you have selected the correct element or used correct action format."
            ret = state.error_message
        elif action["action_name"] in ["stop", "save", "nop"]:  # ok, nothing to do
            ret = f"Browser step: {action_string}"
        elif action["action_name"] == "screenshot":
            _old_mode = state.curr_screenshot_mode
            _fields = action["action_value"].split() + [""] * 2
            _new_mode = _fields[0].lower() in ["1", "true", "yes"]
            _save_path = _fields[1].strip()
            if _save_path:
                try:
                    assert state.boxed_screenshot.strip(), "Screenshot not available!"
                    file_bytes = base64.b64decode(state.boxed_screenshot)
                    _dir = os.path.dirname(_save_path)
                    if _dir:
                        os.makedirs(_dir, exist_ok=True)
                    with open(_save_path, 'wb') as fd:
                        fd.write(file_bytes)
                    save_info = f" (Current screenshot saved to {_save_path}.)"
                except Exception as e:
                    save_info = f" (Error {e} when saving screenshot.)"
            else:
                save_info = ""
            state.curr_screenshot_mode = _new_mode
            ret = f"Browser step: {action_string} -> Changing curr_screenshot_mode from {_old_mode} to {_new_mode}" + save_info
        else:
            # actually perform action
            action_succeed = self.action(state.browser_id, state.page_id, action)
            if not action_succeed:  # no succeed
                state.error_message = f"The action you have chosen cannot be executed: {action_string}. Please double-check if you have selected the correct element or used correct action format."
                ret = state.error_message
            else:  # get new states
                results = self._get_accessibility_tree_results(state)
                state.update(**results)  # update it!
                ret = f"Browser step: {action_string}"
        return ret
        # --

    # sync files between remote and local dirs
    def sync_files(self):
        # --
        def _get_file(_f: str):
            url = f"http://{self.web_ip}/getFile"
            data = {"filename": _f}
            try:
                response = requests.post(url, json=data, timeout=self.web_timeout)
                if response.status_code == 200:
                    res_json = response.json()
                    base64_str = res_json["file"]
                    file_bytes = base64.b64decode(base64_str)
                    if _f:
                        _dir = os.path.dirname(_f)
                        if _dir:
                            os.makedirs(_dir, exist_ok=True)
                    with open(_f, 'wb') as fd:  # Change output filename as needed
                        fd.write(file_bytes)
                    return True
                else:
                    zwarn(f"Get file failed with status code: {response.status_code}")
                    return False
            except Exception as e:
                zwarn(f"Request failed: {e}")
                return False
        # --
        files = {}
        for file in self.state.downloaded_file_path:
            if not os.path.exists(file):
                fres = _get_file(file)
                files[file] = f"Get[res={fres}]"
            else:
                files[file] = "Exist"
        zlog(f"Sync files: {files}")

    def screenshot_mode(self, flag=None):
        old_mode = self.state.curr_screenshot_mode
        new_mode = old_mode
        if flag is not None:  # set as flag
            self.state.curr_screenshot_mode = flag
        return old_mode, new_mode
