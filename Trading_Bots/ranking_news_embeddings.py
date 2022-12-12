import numpy as np 
import pandas as pd 
import random 
import os 
from tqdm.auto import tqdm 
import time 
import datetime
from transformers import * 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from scipy.spatial.distance import cdist 

df = pd.read_csv("full_news_october_1st.csv")
print("shape = {}".format(df.shape)) 

model = AutoModel.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 
tokenizer = AutoTokenizer.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 

titles = df["titles"].values 
contents = df["contents"].values 
years = df["year"].values 
months = df["month"].values 
days = df["day"].values 
hours = df["hour"].values 

input_ids, attention_masks = [], [] 

for i in tqdm(range(len(titles)), position=0, leave=True, desc="tokenzing"): 
    title = str(titles[i]) 
    content = str(contents[i]) 
    encoded_inputs = tokenizer(title, content, max_length=512, padding="max_length", truncation=True)  
    input_ids.append(encoded_inputs["input_ids"]) 
    attention_masks.append(encoded_inputs["attention_mask"]) 
    
input_ids = torch.tensor(input_ids, dtype=int) 
attention_masks = torch.tensor(attention_masks, dtype=int) 

candidate_embeddings = [] 
# model settings 
if torch.cuda.is_available(): 
    print("moving model to GPU") 
    model.cuda() 
else: 
    print("moving model to CPU") 
    # no action required 
model.eval() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

print(f"currently using {device} for computation") 

for i in tqdm(range(len(input_ids)), position=0, leave=True, desc="calculating embeddings"): 
    with torch.no_grad(): 
        input_id = input_ids[i] 
        input_id = torch.reshape(input_id, (-1, 512)) 
        input_id = input_id.to(device) 
        attention_mask = attention_masks[i] 
        attention_mask = torch.reshape(attention_mask, (-1, 512))  
        attention_mask = attention_mask.to(device) 
        embedding = model(input_id, attention_mask)[0][:,0,:] 
        candidate_embeddings.append(embedding) 
        
candidate_embeddings = torch.cat(candidate_embeddings, dim=0) 

print("saving news candidates...") 
torch.save(candidate_embeddings, "news_candidate_embeddings.pt") 
