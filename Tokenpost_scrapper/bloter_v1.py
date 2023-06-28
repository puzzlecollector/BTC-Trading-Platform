import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup, Comment
from dateutil import parser
from lxml import etree

def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "lxml")
    title = soup.find("h1", {"class":"heading"}).text.strip()
    content = soup.find("article", {"id":"article-view-content-div"}).text.strip()
    meta_info = soup.find("article", {"class":"item"})
    meta_str = ""
    for child_div in meta_info.find_all("li"):
        meta_str += child_div.text.strip()
    return title, content, meta_str

titles, contents, metas = [], [], []

for i in tqdm(range(1, 501), position=0, leave=True, desc="pages"):
    try:
        url = f"https://www.bloter.net/news/articleList.html?page={i}&total=50587&box_idxno=&view_type=sm"

        headers = requests.utils.default_headers()
        headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
        })
        news_req = requests.get(url, headers=headers)

        soup = BeautifulSoup(news_req.content,"lxml")
        elems = soup.find_all("section", {"id":"section-list"})

        links = []

        for e in elems:
            for child_div in e.find_all("div", {"class":"view-cont"}):
                for a in child_div.find_all("a"):
                    l = a["href"]
                    links.append("https://www.bloter.net" + l)


        links, indices = np.unique(links, return_index=True)
        sorted_links = links[np.argsort(indices)]

        for curlink in sorted_links:
            try:
                title, content, meta_str = get_articles(headers, curlink)
                titles.append(title)
                contents.append(content)
                metas.append(meta_str)
            except Exception as e:
                print("Error while scraping news content")
                print(e)
            time.sleep(0.1)
    except Exception as e:
        print(e)
        print("error while scraping!")
    time.sleep(0.2)

print("creating dataframe")
df = pd.DataFrame(list(zip(titles, contents, metas)), columns=["titles", "contents", "meta_information"])
df.to_csv("bloter_20230628.csv",index=False)
