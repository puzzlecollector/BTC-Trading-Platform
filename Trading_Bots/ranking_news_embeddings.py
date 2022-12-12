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

df = pd.read_csv("full_news_october_1st.csv")
print("shape = {}".format(df.shape)) 

model = AutoModel.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 
tokenizer = AlbertTokenizer.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 

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

batch_size = 32 
data = TensorDataset(input_ids, attention_masks) 
sampler = SequentialSampler(data) 
dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size) 

for step, batch in tqdm(enumerate(dataloader), desc="calculating embeddings", position=0, leave=True, total=len(dataloader)): 
    batch = tuple(t.to(device) for t in batch) 
    b_input_ids, b_attn_masks = batch 
    with torch.no_grad():
        output = model(b_input_ids, b_attn_masks)[0][:,0,:]
        output = output.detach().cpu() 
        for i in range(output.shape[0]): 
            candidate_embeddings.append(torch.reshape(output[i], (-1, 768))) 

candidate_embeddings = torch.cat(candidate_embeddings, dim=0)  

print(candidate_embeddings.shape) 

print("saving news candidates...") 
torch.save(candidate_embeddings, "news_candidate_embeddings.pt") 
