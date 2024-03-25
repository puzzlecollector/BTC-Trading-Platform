import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import time
import requests
from bs4 import BeautifulSoup
from dateutil import parser


# webscrape coinness news articles from tokenpost
# 2495 page
def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "html.parser")
    # Extracting title
    title = soup.find("h1", {"class": "view_top_title noselect"}).text.strip()
    # Finding the specific <div>
    article_content_div = soup.find('div', class_='article_content', itemprop='articleBody')
    content = ""  # Initialize content as empty string
    # Check if the div was found
    if article_content_div:
        # Extracting text from all <p> tags within the <div>
        p_tags = article_content_div.find_all('p')
        for p in p_tags:
            content += p.get_text(strip=True) + " "  # Appending each <p> content with a space for readability

        # Optionally, remove specific unwanted text
        unwanted_text = "이 광고는 쿠팡 파트너스 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받습니다."
        content = content.replace(unwanted_text, "").strip()
    else:
        content = "No content found in the specified structure."
    return title, content

def scrape_tokenpost():
    all_titles, all_contents, all_full_times = [], [], []
    for i in tqdm(range(1, 2496), desc="Scraping content from tokenpost"):
        try:
            links = []
            url = f"https://www.tokenpost.kr/coinness?page={i}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            news_req = requests.get(url, headers=headers)
            soup = BeautifulSoup(news_req.content, "html.parser")
            elems = soup.find_all("div", {"class": "list_left_item"})
            for e in elems:
                article_elems = e.find_all("div", {"class": "list_item_text"})
                for article in article_elems:
                    title_link = article.find("a", href=True)
                    if title_link and '/article-' in title_link['href']:
                        full_link = 'https://www.tokenpost.kr' + title_link['href']
                        # Find the date element in the parent of the article
                        date_elem = article.parent.find("span", {"class": "day"})
                        news_date = parser.parse(date_elem.text)
                        links.append(full_link)
                        all_full_times.append(news_date)
                    if len(all_full_times) > 4:
                        break
            for link in links:
                try:
                    title, content = get_articles(headers, link)
                    all_titles.append(title)
                    all_contents.append(content)
                except Exception as e:
                    print(f"Error while scraping news content: {e}")
        except Exception as e:
            print(f"Error while scraping page {i}: {e}")
        time.sleep(0.1)

    if len(all_titles) == 0 and len(all_full_times) == 5:
        for k in range(5):
            all_titles.append('')
            all_contents.append('')

    return pd.DataFrame({'titles': all_titles, 'contents': all_contents, 'datetimes': all_full_times})


df = scrape_tokenpost()

print(df)

df.to_csv("coinness_0325.csv", index=False)
