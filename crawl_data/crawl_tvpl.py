import os
import random
import time
import traceback
from bs4 import BeautifulSoup
import pandas as pd
import requests
from tabulate import tabulate
import webbrowser

PAG_URL_FORMATS = {
    "Văn Bản Pháp Luật": "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=0&status=0&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page={0}",
    "Dự Thảo": "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=0&status=100&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page={0}",
    "Công Văn": "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=3&status=0&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page={0}",
    "Tiêu Chuẩn Việt Nam": "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=39&status=0&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page={0}"
}
PAG_PAGE_COUNT = 20
INIT_INDEXS = {
    "Văn Bản Pháp Luật": 0,
    "Dự Thảo": 0,
    "Công Văn": 0,
    "Tiêu Chuẩn Việt Nam": 0
}
PAGE_PARTS = PAG_PAGE_COUNT * 10
BASE_DELAY_TICK = 3000
DELTA_DELAY_TICK = 500
FOLDER_PATH = "datasets"
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
    def __init__(self, base_delay_tick: int = 3000, delta_delay_tick: int = 500):
        self.base_delay_tick = base_delay_tick
        self.delta_delay_tick = delta_delay_tick
        self.next_request_tick = 0

    def get_soup(self, url: str):
        curr_ticks = time.time()
        wait_ticks = self.next_request_tick - curr_ticks
        if wait_ticks > 0:
            time.sleep(wait_ticks * 1000)
            self.next_request_tick = curr_ticks + self.base_delay_tick + \
                random.randint(-self.delta_delay_tick, self.delta_delay_tick)

        response = requests.get(url, headers=REQUEST_HEADER)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        else:
            raise Exception(f"Failed to retrieve {url}. Status code: {
                            response.status_code} \n {response}")


def load_indexs(file_name: str = 'page_indexes.txt'):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            return eval(content)
    except FileNotFoundError:
        return INIT_INDEXS
    except Exception as e:
        print(f"An error occurred while loading the indexes: {e}")
        print("Load init indexs")
        return INIT_INDEXS


def save_indexs(page_indexes: dict[str, int], file_name: str = 'page_indexes.txt'):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(str(page_indexes))


page_indexes = load_indexs()
scraper = WebScraper(BASE_DELAY_TICK, DELTA_DELAY_TICK)


def load_parquet(path: str):
    try:
        df = pd.read_parquet(path)
        print(f"Loaded existing data from {path}")
    except FileNotFoundError:
        df = pd.DataFrame()
        print(f"No existing file found. Created a new DataFrame.")
    return df


def save_parquet(df: pd.DataFrame, path: str):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    df.to_parquet(path, engine='pyarrow', index=False)


def append_parquet(df: pd.DataFrame, id: str, url: str, title: str, created_date: str, updated_date: str, content: str):
    new_data = pd.DataFrame({
        'id': [id],
        'url': [url],
        'title': [title],
        'created_date': [created_date],
        'updated_date': [updated_date],
        'content': [content]
    })
    return pd.concat([df, new_data], ignore_index=True)

def crawl_content(url: str):
    soup = scraper.get_soup(url)
    content = soup.find(id="ctl00_Content_ThongTinVB_pnlDocContent").find(
        class_=["content1"])
    if content.find(class_=["TaiVanBan"]):
        return ("Cần tải pdf", "")
    
    content = soup.find(id="ctl00_Content_ThongTinVB_pnlDocContent").find(class_=["content1"]).prettify().encode('utf-8', 'backslashreplace')
    return ("", content)

def gen_part_path(name, index): return f"{FOLDER_PATH}/{name}_p{index * PAGE_PARTS}-{(index + 1) * PAGE_PARTS - 1}.parquet"
                      
def crawl_data(name: str):
    curr_url = ""
    base_index = page_indexes[name]
    pagination_url_format = PAG_URL_FORMATS[name]

    part_index = base_index // PAGE_PARTS
    pag_index = base_index // PAG_PAGE_COUNT
    curr_index = base_index % PAG_PAGE_COUNT

    parquet_df = load_parquet(gen_part_path(name, part_index))

    try:
        while True:
            soup = scraper.get_soup(
                pagination_url_format.format(pag_index + 1))
            items = soup.find_all(class_=['content-1', 'content-0'])

            for i in range(curr_index, len(items)):
                item = items[i]
                a_element = item.find('p', class_='nqTitle').a
                date_parent = item.find(
                    'div', class_='right-col').find_all('p')
                curr_url = url = a_element['href']
                title = a_element.text.strip()
                created_date = date_parent[0].text.split(':')[-1].strip()
                updated_date = date_parent[-1].text.split(':')[-1].strip()
                (msg, content) = crawl_content(url)

                if content:
                    print(f"\033[92m✅ {base_index}, {url}")
                    parquet_df = append_parquet(parquet_df, base_index, url, title,
                                   created_date, updated_date, content)
                else:
                    print(f"\033[91m❌ {base_index}, {msg}, {url}")

                base_index += 1
                curr_index += 1
                curr_url = ""

            # refresh
            pag_index += 1
            curr_index = 0

            if base_index // PAGE_PARTS > part_index:
                save_parquet(parquet_df, gen_part_path(name, part_index))

                part_index = base_index // PAGE_PARTS
                parquet_df = load_parquet(gen_part_path(name, part_index))

            if len(items) < PAG_PAGE_COUNT:
                print("Reached last page. Completed")
    except KeyboardInterrupt:
        print("Execution was interrupted by the user (Ctrl+C).")
    except Exception as e:
        print("Err: ")
        print(e)
        traceback.print_exc()
        print("At pag: ", pagination_url_format.format(pag_index + 1))
        print("At url: ", curr_url)
    finally:
        def hard_save(): 
            print("On try hard save...")
            try: 
                page_indexes[name] = base_index
                save_indexs(page_indexes)
                save_parquet(parquet_df, gen_part_path(name, part_index))
                print("Saved current state")
            except KeyboardInterrupt:
                print("Execution was interrupted by the user (Ctrl+C).")
                hard_save()
                
        hard_save()

def read_data(name: str):
    while True:
        try: 
            base_index = int(input("Input index: "))  
            part_index = base_index // PAGE_PARTS
            
            parquet_df = load_parquet(gen_part_path(name, part_index))
            matching_rows = parquet_df[parquet_df['id'] == base_index]
            with open("temp.html", "w", encoding='utf-8') as file:
                file.write(matching_rows["content"].iloc[0].decode('utf-8'))
            
            table_data = [
                ["ID", matching_rows["id"].iloc[0]],
                ["Created Date", matching_rows["created_date"].iloc[0]],
                ["Updated Date", matching_rows["updated_date"].iloc[0]],
                ["Title", matching_rows["title"].iloc[0]],
                ["URL", matching_rows["url"].iloc[0]],
            ]
            print(tabulate(table_data, tablefmt="grid"))
            webbrowser.open(f"file://{os.path.abspath('temp.html')}")
        except KeyboardInterrupt:
            print("Execution was interrupted by the user (Ctrl+C).")
            break
        except Exception as e:
            print("Err: ")
            print(e)
            traceback.print_exc()
        
        
# program
name = list(INIT_INDEXS.keys())[int(input(f"Read title ({', '.join([f"{key}: {index}" for index, key in enumerate(INIT_INDEXS.keys())])}): "))]
if input("Crawl(y) or Read(else): ").lower() == "y":
    crawl_data(name)
else:
    read_data(name)
        