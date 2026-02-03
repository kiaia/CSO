from selenium.webdriver.common.by import By
import io
from io import BytesIO
from requests.exceptions import RequestException
import requests
import helium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import re
from helium import start_chrome, go_to, click, write, find_all, S, set_driver, find_all, S, click, scroll_down, \
    scroll_up

from smolagents import tool, Tool
from smolagents.agents import ActionStep
from smolagents.tools import Tool

from helium import get_driver, find_all, S
from PIL import Image
import time
from markdownify import markdownify
# from model_list import automatedModelConstruction
# from openai import AzureOpenAI

from dataclasses import dataclass
from typing import Dict

import mimetypes
# from scripts.text_inspector_tool import TextInspectorTool
# from scripts.cookies import COOKIES_LIST
from .cookies import COOKIES_LIST

# --
ASK_LLM_TOOL = None

def setup_ask_llm_tool(ask_llm_tool):  # ugly...
    global ASK_LLM_TOOL
    ASK_LLM_TOOL = ask_llm_tool

def ask_llm(query):
    return ASK_LLM_TOOL(query)
# --

def save_screenshot(memory_step, agent):
    time.sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        # Remove previous screenshots from logs for lean processing
        for previous_memory_step in agent.memory.steps:
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None

        # Highlight clickable elements
        # highlight_clickable_elements(driver)

        # Take a screenshot
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

        # Update observations with current URL
        url_info = f"Current url: {driver.current_url}"
        memory_step.observations = (
            url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
        )


def highlight_clickable_elements():
    driver = helium.get_driver()

    set_driver(driver)
    # Find all clickable elements
    clickable_elements = find_all(
        S('a, button, input, textarea, select, [role="button"], [tabindex]:not([tabindex="-1"])'))

    # Iterate over the elements, add a border and label them
    for index, element in enumerate(clickable_elements, start=1):
        # Use JavaScript to add a red border and label
        driver.execute_script(
            f"var elem = arguments[0];"
            f"var label = document.createElement('span');"
            f"label.style.color = 'red';"
            f"label.style.fontSize = '10px';"
            f"label.style.position = 'absolute';"
            f"label.style.zIndex = '1000';"
            f"label.style.backgroundColor = 'white';"  # Add background to make the label more visible
            f"label.style.padding = '2px';"  # Add padding for better visibility
            f"label.style.borderRadius = '3px';"  # Add border radius for aesthetics
            f"label.style.right = '-25px';"  # Position the label outside the element
            f"label.style.top = '-25px';"  # Adjust the top position as needed
            f"label.innerText = '{index}';"
            f"elem.style.position = 'relative';"
            f"elem.style.border = '3px solid red';"
            f"elem.appendChild(label);",
            element.web_element
        )

@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    prompt = """
The user is looking for text on the following webpage. Please check if the webpage contains relevant content based on the text provided by the user. The reply format is as follows:

## 1. Paragraphs where the text is located (listed, All)
## 2. All occurrences of the text and the information related to it

## Text
{}
## Web Content
{} 
"""
    driver = helium.get_driver()
    url = driver.current_url
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Convert the HTML content to Markdown
    markdown_content = markdownify(response.text).strip()

    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"# Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"# Focused on element {nth_result} of {len(elements)}\n\n # The basic information is:\n" + ask_llm(prompt.format(text, markdown_content))
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver = helium.get_driver()

    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    driver = helium.get_driver()

    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    driver = helium.start_chrome(headless=False, options=chrome_options)
    # for cookie in COOKIES_LIST:
    #     driver.add_cookie(cookie)
    return driver


def generate_selector(element):
    # Attempt to create a unique selector based on ID, class, or tag
    tag = element.tag_name
    element_id = element.get_attribute('id')
    element_class = element.get_attribute('class')

    if element_id:
        return f"{tag}#{element_id}"
    elif element_class:
        classes = ".".join(element_class.split())
        return f"{tag}.{classes}"
    else:
        return tag


@tool
def generate_a11y_tree() -> str:
    """Parse the webpage's accessibility tree and return the index, description, selector, URL, type, value, etc., for each element. "
    Returns:
        The accessibility tree of the webpage.
    """
    driver = helium.get_driver()

    set_driver(driver)
    selectors = 'a, button, input, textarea, select, [role="button"], [tabindex]:not([tabindex="-1"])'
    elements = find_all(S(selectors))
    index = 1
    have = set()
    tree_str = ""
    number = 300

    for element in elements:
        try:
            web_element = element.web_element
            tag_name = web_element.tag_name.lower()

            element_id = web_element.get_attribute('id')
            element_type = web_element.get_attribute('type')
            element_value = web_element.get_attribute('value')
            element_aria_label = web_element.get_attribute('aria-label')

            description = web_element.text.strip() or web_element.get_attribute('alt') or "No description"
            url = web_element.get_attribute('href') if tag_name == 'a' else ""
            print_name = tag_name
            if tag_name == "a":
                print_name = "link"
            print_name = print_name if element_aria_label is None else element_aria_label

            # 构建节点标识符
            node_identifier = f"{print_name} '{description}'"
            # if url:
            #     node_identifier += f" url: {url}"
            if element_type:
                node_identifier += f" type: {element_type}"
            if element_value:
                node_identifier += f" value: {element_value}"

            if node_identifier not in have:
                index += 1
                have.add(node_identifier)

                if len(have) >= number:
                    continue

                tree_str += f"[{index}] {node_identifier}\n"


        except:
            pass

    return "The accessibility tree is:\n\n" + tree_str


def generate_a11y_tree_json():
    """Parse the webpage's accessibility tree and return the index, description, selector, URL, type, value, etc., for each element. The format of accessibility tree will be like: "{'index': 57, 'description': 'email', 'category': 'a', 'selector': 'a', 'url': 'https://subscribe.sorryapp.com/24846f03/email/new ', 'type': '', 'value': ''}\n {'index': 58, ...}"

    """
    driver = helium.get_driver()

    set_driver(driver)
    # go_to(url)
    selectors = 'a, button, input, textarea, select, [role="button"], [tabindex]:not([tabindex="-1"])'
    elements = find_all(S(selectors))
    a11y_tree = []
    index = 1
    have = set()
    for element in elements:
        try:
            web_element = element.web_element
            tag_name = web_element.tag_name.lower()

            element_id = web_element.get_attribute('id')
            element_type = web_element.get_attribute('type')
            element_value = web_element.get_attribute('value')
            element_aria_label = web_element.get_attribute('aria-label')

            # Create a more specific selector
            selector = f"{tag_name}[type='{element_type}']" if element_type else tag_name
            if element_id:
                selector = f"{tag_name}#{element_id}"

            item = {
                "index": index,
                "description": web_element.text.strip() or web_element.get_attribute('alt') or "No description",
                "category": tag_name,
                "selector": selector,
                "url": web_element.get_attribute('href') if tag_name == 'a' else "",
                "type": element_type or "",
                "value": element_value or ""
            }
            a11y_tree.append(item)
            index += 1
        except:
            pass
    return a11y_tree


@tool
def perform_click(index: int) -> str:
    """
    Click the button or link in the website.
    Args:
        index: The index of the element in the accessibility tree.
    """
    driver = helium.get_driver()
    set_driver(driver)
    a11y_tree = generate_a11y_tree_json()

    item = next((x for x in a11y_tree if x["index"] == index), None)

    if item["url"]:
        go_to(item["url"])
    elif item["selector"]:
        click(S(item["selector"]))
    else:
        return "No valid method to perform click found"
    return "Click maybe successed! Please check it!"


@tool
def perform_input(index: int, text: str, ) -> str:
    """
    Enter the text into the text box corresponding to the index.
    Args:
        index: The index of the text box in the accessibility tree.
        text: The content you want to input into the text box.
    """
    driver = helium.get_driver()

    set_driver(driver)
    a11y_tree = generate_a11y_tree_json()

    item = next((x for x in a11y_tree if x["index"] == index), None)

    if item["category"] in ['input', 'textarea'] and item["type"] not in ['button', 'submit', 'reset']:
        write(text, into=S(item["selector"]))
    else:
        return "Element is not suitable for text input"

    return "Text input maybe successed! Please check it!"


@tool
def scroll_down_window(num_pixels: int) -> None:
    """This will scroll one viewport down.
    Args:
        num_pixels: The number of pixels you want to scroll down."""
    scroll_down(num_pixels=num_pixels)


@tool
def scroll_up_window(num_pixels: int) -> None:
    """This will scroll one viewport down.
    Args:
        num_pixels: The number of pixels you want to scroll up."""
    scroll_up(num_pixels=num_pixels)


# def clean_web_page(web_text: str) -> str:
#     prompt = """
# Here is the plain text information of a webpage, which contains HTML tags as well as useful text content. Please clean up the webpage and extract the useful information. The return format should be:
#
# ## 1. Summary of the Main Content of the Webpage (short).
# ## 2. The complete content of the Webpage (without loss, DO NOT summarize).
#
# The web page:
# """
#     response = client.chat.completions.create(
#         model="gpt-4o",  # model = "deployment_name".
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt + web_text},
#                 ],
#             }
#         ]
#     )
#     return response.choices[0].message.content


def set_cookies(url):
    driver = helium.get_driver()
    # COOKIES_LIST
    for cookie in COOKIES_LIST:
        if cookie['domain'].strip(". ") in url:
            driver.add_cookie({
                "domain": cookie['domain'],
                "name": cookie['name'],
                "value": cookie['value']})
    driver.refresh()


@tool
def visit_webpage(url: str, ) -> str:
    """Markdownify the webpage you visited to provide basic webpage information. Provide the accessibility tree for more precise web-related action.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown and the accessibility tree, or an error message if the request fails.
    """
    info = ""
    try:
        # Send a GET request to the URL
        go_to(url)
        set_cookies(url)
        info += f"Successfully Visited the URL: {url}.\n"
    except Exception as e:
        return info + f"However, An unexpected error occurred: {str(e)}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # llm_cleaned_web = clean_web_page(response.text)
        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return info + "Webpage content:\n\n" + markdown_content[:3000] + "\n\n" + generate_a11y_tree()

    except RequestException as e:
        return info + f"However, Error fetching the text content of webpage due to: {str(e)}\n Read the webpage by screenshot instead."
    except Exception as e:
        return info + f"However, An unexpected error occurred: {str(e)}"


@dataclass
class PreTool:
    name: str
    inputs: Dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class MixedSearchTool(Tool):
    name = "web_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=10, **kwargs):
        super().__init__()
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
            from googlesearch import search
        except ImportError as e:
            raise ImportError(
                "You must install package `duckduckgo_search` and googlesearch-python to run this tool: for instance run `pip install duckduckgo-search`."
            ) from e
        self.ddgs = DDGS(**kwargs)
        self.google_search = search

    def forward(self, query: str) -> str:
        results = self.ddgs.text(query, max_results=self.max_results)
        url_set = set([result['href'] for result in results])
        try:
            google_results = self.google_search(query, advanced=True, num_results=self.max_results, )
            # google_results=[]
        except:
            print("REACH THE LIMITATION OF GOOGLE SEARCH")
            google_results = []
        for google_result in google_results:
            if google_result.url not in url_set:
                # avoid repeatation
                results.append(
                    {"title": google_result.title, "href": google_result.url, "body": google_result.description})
                url_set.add(google_result.url)
        prompt = """Here is a query, along with the Google search results. Please reorder the search results based on their relevance to the query.
## Query
{}
## Search results
{}"""
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]

        query = prompt.format(query, "## Search Results\n\n" + "\n\n".join(postprocessed_results))
        llm_processed_results = ask_llm(query)
        return llm_processed_results


class DownloadTool(Tool):
    name = "download_file"
    description = """
Download a file at a given URL (not a path in the computer). The file should be of this format: [".txt",".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".png", ".docx"]
After using this tool, for further inspection of this page you should return the download path to your manager via final_answer, and they will be able to inspect it.
DO NOT use this tool for .htm or .pdf files: for these types of files use visit_page with the file url instead."""
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the file to be downloaded."},
              "file_format": {"type": "string", "description": "Optional: The format of the file to be downloaded.",
                              "nullable": True}}
    output_type = "string"

    def __init__(self, ):
        super().__init__()
        self.item = 0
        # self.browser = browser

    def forward(self, url: str, file_format: str = None) -> str:
        if "arxiv" in url:
            url = url.replace("abs", "pdf")
        response = requests.get(url)
        # response = requests.get(pdf_url, headers=send_headers)
        bytes_io = io.BytesIO(response.content)
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type)
        # ind = random.choice()
        if file_format:
            extension = file_format
        elif url.endswith(".txt"):
            extension = ".txt"
        elif extension is None and "pdf" in url:
            extension = ".pdf"
        else:
            extension = ".pdf"

        if "htm" or "pdf" in extension:
            raise Exception("Do not use this tool for html and pdf files: use visit_page instead.")

        if extension and isinstance(extension, str):
            new_path = f"./downloads/file-{self.item}{extension}"
        else:
            new_path = f"./downloads/file-{self.item}.object"
        self.item += 1

        if url.endswith(".pdf") or ".pdf" in url:
            with open(new_path, mode='wb') as f:
                f.write(bytes_io.getvalue())
        else:
            with open(new_path, "wb") as f:
                f.write(response.content)

        # try:
        #     model = automatedModelConstruction("gpt-4o")
        #
        #     text_limit = 100000
        #     ti_tool = TextInspectorTool(model, text_limit)
        #     description = ti_tool.forward(new_path)
        # except:
        #     raise Exception(
        #         "The tool can ONLY download files that have explicit file suffixes in the URL. If the URL does not have a suffix, it may be necessary to open it in a browser for rendering before downloading.")
        # return f"File was downloaded and saved under path {new_path}. {description}"

        return f"File was downloaded and saved under path {new_path}."
