# example of scraping news dataset from Tokenpost 
# scraping news from the first 250 pages 
import time
import ccxt
import numpy as np
import pandas as pd
import telegram
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup, Comment
from dateutil import parser

def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "lxml")
    title = soup.find("p",{"class":"ArticleBigTitle"}).text.strip()
    content = soup.find("div", {"class":"viewArticle"}).text.strip()
    return title, content

titles, contents, full_times = [], [], []
for i in tqdm(range(1, 250), position=0, leave=True):
    try:
        links, times = [], []
        url = "https://www.tokenpost.kr/coinness?page=" + str(i)
        headers = requests.utils.default_headers()
        headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
        })
        news_req = requests.get(url, headers=headers)
        soup = BeautifulSoup(news_req.content,"lxml")
        elems = soup.find_all("div", {"class":"listWrap"})
        for e in elems:
            for child_div in e.find_all("div", {"class":"articleListWrap paddingT15 paddingB15"}):
                for a in child_div.find_all("a"):
                    l = a['href']
                    if '/article-' in l:
                        links.append('https://www.tokenpost.kr' + str(l))
            for child_div in e.find_all("span", {"class":"articleListDate marginB8"}):
                news_dt = parser.parse(child_div.text)
                times.append(news_dt)

        for t in times:
            full_times.append(t)

        for idx, date in enumerate(times):
            try:
                title, content = get_articles(headers, links[idx])
                titles.append(title)
                contents.append(content)
            except Exception as e:
                print("Error occurred with getting article content!")
                print(e)
            time.sleep(0.2)
    except Exception as e:
        print(e)
        print("Error while scraping!")
    time.sleep(0.2)

print("Creating additional news dataFrame...")
df = pd.DataFrame(list(zip(titles, contents, full_times)),
               columns =['titles', 'contents', 'datetimes'])
df.to_csv("additional_tokenpost.csv", index=False)
