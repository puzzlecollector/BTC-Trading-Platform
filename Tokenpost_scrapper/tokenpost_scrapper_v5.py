# import time
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser
import time

def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "html.parser")
    title = soup.find("span", {"class": "view_top_title noselect"}).text.strip()
    content = soup.find("div", {"class": "view_text noselect"}).get_text(strip=True)
    return title, content

def scrape_tokenpost():
    all_titles, all_contents, all_full_times = [], [], []
    for i in tqdm(range(1, 2), desc="Scraping content from tokenpost"):
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
    return pd.DataFrame({'titles': all_titles, 'contents': all_contents, 'datetimes': all_full_times})

# Run the scraping and save to CSV
start = time.time() 
df = scrape_tokenpost()
elapsed = time.time() - start 
print(elapsed) 
