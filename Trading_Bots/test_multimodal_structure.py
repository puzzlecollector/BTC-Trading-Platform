'''
testing a multimodal structure 
'''

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
from transformers import AutoModel, AlbertTokenizer, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import time
import math
from datetime import datetime 
import pickle

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, dropout=0.1, max_len=5000): 
        super(PositionalEncoding, self).__init__() 
        self.dropout = nn.Dropout(p=dropout) 
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0,1) 
        self.device = torch.device("cuda") 
        self.register_buffer("pe", pe)  
    def forward(self, x): 
        x = x.to(self.device) 
        self.pe = self.pe.to(self.device) 
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x) 

class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout 
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples 
    def forward(self, out): 
        return torch.mean(torch.stack([self.classifier(self.dropout(p=self.max_dropout_rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)

class AttentivePooling(torch.nn.Module): 
    def __init__(self, input_dim): 
        super(AttentivePooling, self).__init__() 
        self.W = nn.Linear(input_dim, 1) 
    def forward(self, x): 
        softmax = F.softmax 
        att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1) 
        x = torch.sum(x * att_w, dim=1) 
        return x 
    
class MultiModalTrader(nn.Module):
    def __init__(self, plm, chart_features, sequence_length, d_model, num_classes, n_heads, num_encoders, news_count, num_input_tokens): 
        super(MultiModalTrader, self).__init__()         
        ## chart related variables ## 
        self.chart_features = chart_features 
        self.sequence_length = sequence_length 
        self.d_model = d_model 
        self.num_classes = num_classes 
        self.n_heads = n_heads 
        self.num_encoders = num_encoders 
        self.chart_embedder = nn.Sequential(
            nn.Linear(self.chart_features, d_model//2), 
            nn.ReLU(), 
            nn.Linear(d_model//2, d_model) 
        ) 
        self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_encoders) 
        self.attentive_pooling = AttentivePooling(input_dim=self.d_model) 
        
        ## news related variables ## 
        self.news_count = news_count 
        self.num_input_tokens = num_input_tokens 
        self.config = AutoConfig.from_pretrained(plm) 
        self.LM = AutoModel.from_pretrained(plm) 
        self.news_pos_encoder = PositionalEncoding(d_model=self.config.hidden_size) 
        self.news_encoder_layers = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=self.n_heads, batch_first=True) 
        self.news_transformer_encoder = nn.TransformerEncoder(self.news_encoder_layers, num_layers=self.num_encoders) 
        self.news_attentive_pooling = AttentivePooling(input_dim=self.config.hidden_size) 
        
        ## concatenation related variables ## 
        self.fc = nn.Linear(self.d_model + self.config.hidden_size, self.num_classes) 
        self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc) 
    
    def forward(self, chart_seq, news_input_ids, news_attn_masks):
        '''
        chart_seq: [batch_size, sequence_length, num features] 
        news_input_ids: [batch_size, sequence_length, num tokens] 
        news_attention_masks: [batch_size, sequence_length, num tokens] 
        '''
        chart_v = self.chart_embedder(chart_seq) 
        print(chart_v) 
        chart_v = self.pos_encoder(chart_v) 
        chart_v = self.transformer_encoder(chart_v) 
        chart_v = self.attentive_pooling(chart_v) 
        
        batch_size = news_input_ids.size(0) 
        news_v = torch.zeros((batch_size, self.news_count, self.config.hidden_size)) 
        for i in range(batch_size): 
            for j in range(self.news_count): 
                input_id = news_input_ids[i][j] 
                input_id = torch.reshape(input_id, (-1, self.num_input_tokens)) 
                attention_mask = news_attn_masks[i][j] 
                attention_mask = torch.reshape(attention_mask, (-1, self.num_input_tokens)) 
                output = self.LM(input_id, attention_mask)[0][:,0,:] 
                news_v[i][j] = output 
        news_v = self.news_pos_encoder(news_v) 
        news_v = self.news_transformer_encoder(news_v) 
        news_v = self.news_attentive_pooling(news_v) 
        embedding = torch.cat((chart_v, news_v), dim=1) 
        x = self.multi_dropout(embedding) 
        return x 
    
model = MultiModalTrader(plm="totoro4007/cryptodeberta-base-all-finetuned", 
                         chart_features=15, 
                         sequence_length=15, 
                         d_model=256, 
                         num_classes=2, 
                         n_heads=8, 
                         num_encoders=6, 
                         news_count=20, 
                         num_input_tokens=256)
model = model.cuda() 
device = torch.device("cuda") 

chart_seq = torch.zeros((8, 15, 15))
news_input_ids = torch.zeros((8, 20, 256), dtype=int) 
news_attention_masks = torch.zeros((8, 20, 256), dtype=int) 


chart_seq = chart_seq.to(device)
news_input_ids = news_input_ids.to(device) 
news_attention_masks = news_attention_masks.to(device) 

output = model(chart_seq, news_input_ids, news_attention_masks) 


print(output) 
