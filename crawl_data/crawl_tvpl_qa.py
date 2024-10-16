import json
import os
import random
import re
import time
import traceback
import unicodedata
import warnings
from bs4 import BeautifulSoup
import requests

from bs4 import MarkupResemblesLocatorWarning

# Suppress only the specific warning from BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

INIT_INDEX = 200000
END_INDEX = 0

PAG_ANSWER_COUNT = 50
PAG_THREAD_COUNT = 10000
BASE_URL = "https://thuvienphapluat.vn/cong-dong-dan-luat"
API_THREAD_URL = "https://dlapi.thuvienphapluat.vn/thread-api/thread-detail-v3?thread_id={0}&token="
API_ANSWER_URL = "https://dlapi.thuvienphapluat.vn/thread-api/get-post-by-thread?thread_id={0}&user_id=-1&page={1}&pageSize={2}&token="
BASE_DELAY_TICK = 500
ADD_DELAY_TICK = 500
FOLDER_PATH = "qa_datasets"
STATE_FILENAME = 'thread_index.txt'
REQUEST_HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Connection': 'keep-alive',
    'DNT': '1',  # Do Not Track Request Header
    'Upgrade-Insecure-Requests': '1',
    'Referer': 'https://www.google.com/',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}


class WebScraper:
    def __init__(self, base_delay_tick: int = 3000, add_delay_tick: int = 500):
        self.base_delay_tick = base_delay_tick
        self.add_delay_tick = add_delay_tick
        self.next_request_tick = 0

    def get_data(self, url: str):
        curr_ticks = time.time()
        wait_ticks = self.next_request_tick - curr_ticks
        if wait_ticks > 0:
            time.sleep(wait_ticks * 1000)
            self.next_request_tick = curr_ticks + self.base_delay_tick + \
                random.randint(0, self.add_delay_tick)

        response = requests.get(url, headers=REQUEST_HEADER)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to retrieve {url}. Status code: {
                            response.status_code} \n {response}")


class State:
    def __init__(
        self,
        state_filename: str,
        data_folder: str,
        pag_thread_count: int,
        step: int = 1
    ):
        self.state_filename = state_filename
        self.data_folder = data_folder
        self.pag_thread_count = pag_thread_count
        self.step = step

        self.__index = self.__load_index__()
        self.__list = self.__load_list__()

    def __load_index__(self):
        try:
            with open(self.state_filename, 'r', encoding='utf-8') as file:
                content = file.read()
                return int(content)

        except FileNotFoundError:
            return INIT_INDEX

        except Exception as e:
            print(f"An error occurred while loading the indexes: {e}")
            print("Load init indexs")
            return INIT_INDEX

    def __load_list__(self):
        path = self.__gen_part_path__()
        list = []
        try:
            with open(path, 'r', encoding='utf-8') as file:
                # Parse each line as a JSON object (assuming each line is a valid JSON string)
                list = [json.loads(line.strip()) for line in file]

            print(f"Loaded existing data from {path}")
        except FileNotFoundError:
            print(f"No existing file found. Created a new list.")

        return list

    def __gen_part_path__(self):
        part_index = self.__index // self.pag_thread_count
        return f"{FOLDER_PATH}/thread_p{part_index * self.pag_thread_count}-{(part_index + 1) * self.pag_thread_count - 1}"

    def __save_index__(self):
        with open(self.state_filename, 'w', encoding='utf-8') as file:
            file.write(str(self.__index))

    def __save_list__(self):
        path = self.__gen_part_path__()
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'w', encoding='utf-8') as file:
            # Write each dictionary as a separate JSON line
            for item in self.__list:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')

    def inc_index(self):
        if (self.__index + 1) // self.pag_thread_count != self.__index // self.pag_thread_count:
            self.__save_list__()
            self.__index += self.step
            self.__list = self.__load_list__()
        else:
            self.__index += self.step

    def curr_index(self):
        return self.__index

    def add_row(self, data: dict):
        self.__list.append(data)

    def hard_save(self):
        print("On try hard save...")
        try:
            self.__save_index__()
            self.__save_list__()
            print("Saved current state")
        except KeyboardInterrupt:
            print("Execution was interrupted by the user (Ctrl+C).")
            self.hard_save()


class URLFriendlyConverter:
    def __init__(self, replacements: dict[str, str]):
        self.replacements = replacements

    def convert(self, text):
        for old, new in self.replacements.items():
            text = text.replace(old, new)

        text = ''.join(
            (c for c in unicodedata.normalize('NFD', text)
             if unicodedata.category(c) != 'Mn')
        )

        text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
        text = text.replace(' ', '-')
        text = re.sub(r'-+', '-', text)
        text = text.lower()

        return text


state = State(STATE_FILENAME, FOLDER_PATH, PAG_THREAD_COUNT, step=-1)
scraper = WebScraper(BASE_DELAY_TICK, ADD_DELAY_TICK)
url_converter = URLFriendlyConverter({
    'đ': 'd', 'Đ': 'd',
    'ý': 'y', 'Ý': 'y'
})


def get_text(html_like: str):
    return BeautifulSoup(html_like, "html.parser").get_text().strip()


def get_time(time_str: str):
    return int(re.search(r'\d+', time_str).group())


def crawl_answer(id: int):
    result = []
    pag_index = 1

    while True:
        api_url = API_ANSWER_URL.format(id, pag_index, PAG_ANSWER_COUNT)
        data = scraper.get_data(api_url)
        if data["Status"] != True:
            return result

        for item in data["Data"]["Items"]:
            result.append({
                "body": get_text(item["Body"]),
                "post_time": get_time(item["PostDate"]),
                "is_lawyer": item["IsLawyer"],
                "total_thanks": 0 if item["TotalThanks"] is None else item["TotalThanks"]
            })

        if pag_index == 1:
            total_comments = data['Data']['Total']
            print(f"{total_comments} answers, ", end='', flush=True)
            if total_comments > PAG_ANSWER_COUNT:
                print(f"{total_comments // PAG_ANSWER_COUNT} extra loadings, ", end='', flush=True)
        else:
            print(f"{pag_index - 1}, ", end='', flush=True)
            
        if len(data["Data"]["Items"]) < PAG_ANSWER_COUNT:
            return result

        pag_index += 1


def crawl_content(id: int):
    print(f"\033[97m{id}, ", end='', flush=True)
    api_url = API_THREAD_URL.format(id)
    
    try: 
        data = scraper.get_data(api_url)
    except:
        print(f"\033[91m❌, status: 500")
        return
        
    if data["Status"] != True:
        print(f"\033[91m❌, msg: {data["Message"]}")
        return
    
    data = data["Data"]
    title = data["MainPost"]["Subject"]
    url = BASE_URL + "/" + \
        url_converter.convert(title) + "-" + str(id) + ".aspx"
    
    print(f"{url}, ", end='', flush=True)
    answers = crawl_answer(id)

    if len(answers) == 0:
        print(f"\033[91m❌, crawl 0 answer")
        return
    
    if not answers[0]['is_lawyer'] and answers[0]['total_thanks'] < 3:
        print(f"\033[91m❌, pass")
        return
    else:
        answers = [answers[0]]

    result = {
        "post_time": get_time(data["MainPost"]["PostDate"]),
        "title": title,
        "question": get_text(data["MainPost"]["Body"]),
        "tags": [i["Name"].strip() for i in data["Categorie"]],
        "url": url,
        "relevant_topics": [{'title': i["Title"], 'url': BASE_URL + i["Link"]} for i in data["RelatedPosts"]],
        "answers": answers
    }

    print(f"\033[92m✅ PASS")
    return result


def crawl():
    try:
        while state.curr_index() > END_INDEX:
            content = crawl_content(state.curr_index())
            if content != None:
                state.add_row(content)

            state.inc_index()

    except KeyboardInterrupt:
        print("Execution was interrupted by the user (Ctrl+C).")
    except Exception as e:
        print("Err: ")
        print(e)
        traceback.print_exc()
        print("At id: ", state.curr_index())
    finally:
        state.hard_save()


# program
crawl()

# with open('temp.json', 'w', encoding='utf-8') as file:
#   file.write(json.dumps(crawl_content(127), ensure_ascii=False))
