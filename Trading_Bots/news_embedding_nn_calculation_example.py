import numpy as np 
import pandas as pd 
import json
import ccxt 
from tqdm.auto import tqdm
import pandas_ta as ta
import seaborn as sns
from xgboost import XGBClassifier  
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score 
import lightgbm as lgbm  

# import libraries for NN 
import random 
import torch 
from torch import Tensor 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler, IterableDataset  
from tqdm.auto import tqdm  
from transformers import AutoModel, AutoTokenizer, AlbertTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import time
import math
from datetime import datetime, timedelta


# we need to create a module that will collect all the news article collected within some specified four hour period  
news_df = pd.read_csv("full_news_october_1st.csv") 
news_df.head(3)


news_df_dates = [] 

news_years = news_df["year"].values 
news_months = news_df["month"].values 
news_days = news_df["day"].values 
news_hours = news_df["hour"].values 

for i in tqdm(range(len(news_years)), position=0, leave=True): 
    date_str = f"{news_years[i]}-{news_months[i]}-{news_days[i]} {news_hours[i]}:00:00" 
    news_df_dates.append(date_str) 
    
news_df_datetime_objects = [] 
for i in tqdm(range(len(news_df_dates)), position=0, leave=True): 
    date_object = datetime.strptime(news_df_dates[i], '%Y-%m-%d %H:%M:%S')
    news_df_datetime_objects.append(date_object) 
    
    
titles = news_df["titles"].values 
contents = news_df["contents"].values 



# get chart data 
with open('BTC_USDT-15m-4.json') as f:
    d = json.load(f)
    
chart_df = pd.DataFrame(d)
chart_df = chart_df.rename(columns={0:"timestamp",
                                    1:"open",
                                    2:"high",
                                    3:"low",
                                    4:"close",
                                    5:"volume"})

def process(df): 
    binance = ccxt.binance() 
    dates = df['timestamp'].values 
    timestamp = [] 
    for i in range(len(dates)): 
        date_string = binance.iso8601(int(dates[i])) 
        date_string = date_string[:10] + " " + date_string[11:-5] 
        timestamp.append(date_string) 
    df['datetime'] = timestamp 
    df = df.drop(columns={'timestamp'})
    return df

chart_df = process(chart_df)

minutes = [] 
hours = []
days = [] 
months = [] 
years = [] 
for dt in tqdm(chart_df['datetime']): 
    minute = pd.to_datetime(dt).minute 
    hour = pd.to_datetime(dt).hour 
    day = pd.to_datetime(dt).day 
    month = pd.to_datetime(dt).month 
    year = pd.to_datetime(dt).year  
    minutes.append(minute) 
    hours.append(hour) 
    days.append(day) 
    months.append(month)
    years.append(year) 

chart_df["minute"] = minutes 
chart_df['hour'] = hours
chart_df['day'] = days 
chart_df['month'] = months 
chart_df['year'] = years 

datetimes = chart_df["datetime"].values 
chart_minutes = chart_df["minute"].values 
chart_hours = chart_df["hour"].values 
chart_days = chart_df["day"].values 
chart_months = chart_df["month"].values
chart_years = chart_df["year"].values 

chart_df_datetime_objects = [] 
    
for j in range(len(chart_minutes)): 
    chart_date_str = f"{chart_years[j]}-{chart_months[j]}-{chart_days[j]} {chart_hours[j]}:{chart_minutes[j]}:00" 
    chart_date_obj = datetime.strptime(chart_date_str, '%Y-%m-%d %H:%M:%S') 
    chart_df_datetime_objects.append(chart_date_obj) 

all_titles, all_contents = [], [] 
for j in tqdm(range(len(chart_df_datetime_objects))): 
    start_dt = chart_df_datetime_objects[j] 
    end_dt = start_dt + timedelta(hours=4) 
    cur_titles, cur_contents = [], [] 
    for k in range(len(news_df_datetime_objects)): 
        news_dt = news_df_datetime_objects[k] 
        if start_dt <= news_dt and news_dt <= end_dt: 
            cur_titles.append(titles[k]) 
            cur_contents.append(contents[k])  
        if news_dt > end_dt: 
            break 
    all_titles.append(cur_titles) 
    all_contents.append(cur_contents) 
        

L = 100 

for i in range(len(all_titles)): 
    cur_titles = all_titles[i] 
    cur_contents = all_contents[i] 
    while len(cur_titles) < L:
        cur_titles.append("") 
        cur_contents.append("")  
        
    all_titles[i] = cur_titles 
    all_contents[i] = cur_contents
    
    
# evaluation mode 
device = torch.device("cuda") 

tokenizer = AutoTokenizer.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned")  
LM = AutoModel.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 
LM.eval() 
LM.cuda() 

all_embeddings = [] 
for i in tqdm(range(len(all_titles))): 
    embeddings = [] 
    for j in range(len(all_titles[i])): 
        title = all_titles[i][j] 
        content = all_contents[i][j] 
        encoded_inputs = tokenizer(title, content, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device) 
        with torch.no_grad(): 
            output = LM(**encoded_inputs)[0][:,0,:] 
            output = output.detach().cpu().numpy() 
            embeddings.append(output) 
    embeddings = torch.tensor(embeddings) 
    embeddings = torch.squeeze(embeddings, dim=1) 
    all_embeddings.append(embeddings)

all_embeddings = torch.stack(all_embeddings, dim=0) 
print(all_embeddings.shape) 

print("saving extracted embeddings...") 
torch.save(all_embeddings, "all_news_embeddings.pt") 
print("done saving!") 
