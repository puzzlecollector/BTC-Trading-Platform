import numpy as np 
import pandas as pd 
import random 
import os 
from tqdm.auto import tqdm 
import time 
import datetime
from torch.utils.data import *
from transformers import * 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from scipy.spatial.distance import cdist  
import logging 
import re 

def set_global_logging_level(level=logging.ERROR, prefices=[""]): 
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })') 
    for name in logging.root.manager.loggerDict: 
        if re.match(prefix_re, name): 
            logging.getLogger(name).setLevel(level) 

set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"]) 

# load data and models 
df = pd.read_csv("full_news_october_1st.csv")
print("shape = {}".format(df.shape)) 
titles = df["titles"].values 
contents = df["contents"].values 
years = df["year"].values 
months = df["month"].values 
days = df["day"].values 
hours = df["hour"].values 

candidates = torch.load("news_candidate_embeddings.pt")
tokenizer = AlbertTokenizer.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned")
embedding_model = AutoModel.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 
sentiment_model = AutoModelForSequenceClassification.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned")   


def get_embeddings(title, content):  
    embedding_model.eval() 
    encoded_inputs = tokenizer(title, content, max_length=512, padding="max_length", truncation=True, return_tensors="pt") 
    with torch.no_grad(): 
        embedding = embedding_model(**encoded_inputs)[0][:,0,:] 
    return embedding 

def get_sentiment_score(title, content): 
    sentiment_model.eval() 
    encoded_inputs = tokenizer(title, content, max_length=512, padding="max_length", truncation=True, return_tensors="pt") 
    with torch.no_grad(): 
        sentiment = sentiment_model(**encoded_inputs)[0] 
        sentiment = nn.Softmax(dim=1)(sentiment)[0]
    return sentiment 

'''
# example query 
q_id = 5

query = torch.reshape(candidates[q_id],(-1,768))   

sim_scores = cdist(query, candidates, "cosine")[0]

print("="*30 + " " + "query" + " " + "="*30) 
print(titles[q_id]) 
print(contents[q_id]) 
sentiment_score = get_sentiment_score(str(titles[q_id]), str(contents[q_id]))  
print(f"비트코인 가격에 대한 딥러닝 기반 감성 분석 - 호재:{round(sentiment_score[0].item()*100,2)}%, 악재:{round(sentiment_score[1].item()*100,2)}%, 중립:{round(sentiment_score[2].item()*100,2)}%")
print()
print()
print()

topk, cnt = 10, 0 
ranks = np.argsort(sim_scores)
print("="*30 + " " + "candidates" + " " + "="*30) 
for i in range(len(ranks)):  
    if q_id == ranks[i]: 
        continue
    else:
        print(titles[ranks[i]])
        print(contents[ranks[i]]) 
        datestr = str(years[i]) + "년" + str(months[i]) + "월" + str(days[i]) + "일" + str(hours[i]) + "시"
        print(f"datetime: {datestr}")
        print(f"딥러닝 기반 기사 유사도: {1 - sim_scores[ranks[i]]}") 
        sentiment_score = get_sentiment_score(str(titles[ranks[i]]), str(contents[ranks[i]]))  
        print(f"비트코인 가격에 대한 딥러닝 기반 감성 분석 - 호재:{round(sentiment_score[0].item()*100,2)}%, 악재:{round(sentiment_score[1].item()*100,2)}%, 중립:{round(sentiment_score[2].item()*100,2)}%")
        print("="*100) 
        cnt += 1 
    if cnt == topk: 
        break 
''' 
        
        
# example query 2 
q_id = -1 
title = "LINK 고래, 150여개 주소 통해 106만 LINK 스테이킹"
content = "코인데스크에 따르면 오픈씨에서 '올드화이트(Oldwhite)'라고 표시된 고래 주소가 12월 3일부터 7일까지 Aave에서 106만 LINK를 인출, 150여개 주소로 분산 이동한 뒤 스테이킹했다. 앞서 체인링크가 출시한 스테이킹 서비스는 이용자당 스테이킹 한도가 최대 7000 LINK로 제한돼 있다." 

q_emb = get_embeddings(title, content)
sim_scores = cdist(q_emb, candidates, "cosine")[0]

print("="*30 + " " + "query" + " " + "="*30) 
print(title) 
print(content) 
sentiment_score = get_sentiment_score(str(title), str(content))  
print(f"비트코인 가격에 대한 딥러닝 기반 감성 분석 - 호재:{round(sentiment_score[0].item()*100,2)}%, 악재:{round(sentiment_score[1].item()*100,2)}%, 중립:{round(sentiment_score[2].item()*100,2)}%")
print()
print()
print()

topk, cnt = 10, 0 
ranks = np.argsort(sim_scores)
print("="*30 + " " + "candidates" + " " + "="*30) 
for i in range(len(ranks)):  
    if q_id == ranks[i]: 
        continue
    else:
        print(titles[ranks[i]])
        print(contents[ranks[i]]) 
        datestr = str(years[i]) + "년" + str(months[i]) + "월" + str(days[i]) + "일" + str(hours[i]) + "시"
        print(f"datetime: {datestr}")
        print(f"딥러닝 기반 기사 유사도: {1 - sim_scores[ranks[i]]}") 
        sentiment_score = get_sentiment_score(str(titles[ranks[i]]), str(contents[ranks[i]]))  
        print(f"비트코인 가격에 대한 딥러닝 기반 감성 분석 - 호재:{round(sentiment_score[0].item()*100,2)}%, 악재:{round(sentiment_score[1].item()*100,2)}%, 중립:{round(sentiment_score[2].item()*100,2)}%")
        print("="*100) 
        cnt += 1 
    if cnt == topk: 
        break 


