# example of scraping news dataset from Tokenpost
# scraping news from the first 250 pages
import time
import ccxt
import numpy as np
import pandas as pd
import telegram
from tqdm import tqdm
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup, Comment
from dateutil import parser
from lxml import etree

def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "lxml")
    title = soup.find("span",{"class":"view_top_title noselect"}).text.strip()
    #content = soup.find("div", {"class":"view_content_item"}).text.strip()
    dom = etree.HTML(str(soup))
    content = dom.xpath('//*[@id="articleContentArea"]/div[4]/div[1]/p/text()')[0]
    return title, content



titles, contents, full_times = [], [], []
for i in tqdm(range(1, 464), position=0, leave=True, desc="scraping content from tokenpost"):
    try:
        links, times = [], []
        url = "https://www.tokenpost.kr/coinness?page=" + str(i)
        headers = requests.utils.default_headers()
        headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
        })
        news_req = requests.get(url, headers=headers)
        soup = BeautifulSoup(news_req.content,"lxml")
        elems = soup.find_all("div", {"class":"list_left_item"})
        for e in elems:
            for child_div in e.find_all("div", {"class":"list_left_item_article"}):
                for a in child_div.find_all("a"):
                    l = a['href']
                    if '/article-' in l:
                        links.append('https://www.tokenpost.kr' + str(l))
            for child_div in e.find_all("span", {"class":"day"}):
                news_dt = parser.parse(child_div.text)
                times.append(news_dt)

        for t in times:
            full_times.append(t)

        unique_links = np.unique(links)

        for j in range(len(unique_links)):
            try:
                title, content = get_articles(headers, unique_links[j])
                titles.append(title)
                contents.append(content)
            except Exception as e:
                print("Error while scraping news content")
                print(e)
            time.sleep(0.2)

    except Exception as e:
        print(e)
        print("Error while scraping!")
    time.sleep(0.2)

print("Creating additional news dataFrame...")
df = pd.DataFrame(list(zip(titles, contents, full_times)),
               columns =['titles', 'contents', 'datetimes'])
df.to_csv("tokenpost_1_16_3_10.csv", index=False)
