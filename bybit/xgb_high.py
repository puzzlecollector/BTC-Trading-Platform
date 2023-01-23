'''
High XGBoost 

[0]     validation_0-logloss:0.65723
[20]    validation_0-logloss:0.63545
[40]    validation_0-logloss:0.64432
[60]    validation_0-logloss:0.65214
[80]    validation_0-logloss:0.66333
[100]   validation_0-logloss:0.66840
[120]   validation_0-logloss:0.67948
[140]   validation_0-logloss:0.68754
[160]   validation_0-logloss:0.69552
[180]   validation_0-logloss:0.71026
[199]   validation_0-logloss:0.71777
accuracy = 61.59052453468698%
f1 score = 0.614474761843183
'''
import numpy as np 
import pandas as pd 
import json 
import ccxt 
from tqdm.auto import tqdm
import pandas_ta as ta 
import seaborn as sns 
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import f1_score
import random 
import torch
from torch import Tensor 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler 
from transformers import * 
import matplotlib.pyplot as plt 
import time 
import math 
import os 
from xgboost import XGBClassifier  
import joblib 

device = torch.device("cuda:0") 

def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return np.sum(pred_flat==labels_flat) / len(labels_flat) 

### define model ### 
class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
                super(PositionalEncoding, self).__init__() 
                self.dropout = nn.Dropout(p=dropout) 
                pe = torch.zeros(max_len, d_model) 
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
                pe[:, 0::2] = torch.sin(position * div_term) 
                pe[:, 1::2] = torch.cos(position * div_term) 
                pe = pe.unsqueeze(0).transpose(0, 1) 
                self.register_buffer("pe", pe) 
        def forward(self, x):
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
                return torch.mean(torch.stack([self.classifier(self.dropout(p=rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)

class AttentivePooling(nn.Module):
        def __init__(self, input_dim):
                super(AttentivePooling, self).__init__() 
                self.W = nn.Linear(input_dim, 1) 
        def forward(self, x):
                softmax = F.softmax
                att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1) 
                x = torch.sum(x * att_w, dim=1) 
                return x 

class BTC_CLF(nn.Module): 
        def __init__(self, chart_features, sequence_length, d_model, num_classes, n_heads, num_encoders):
                super(BTC_CLF, self).__init__() 
                self.chart_features = chart_features
                self.sequence_length = sequence_length 
                self.d_model = d_model 
                self.num_classes = num_classes 
                self.n_heads = n_heads 
                self.num_encoders = num_encoders 
                self.batchnorm = nn.BatchNorm1d(sequence_length) 
                self.chart_embedder = nn.Sequential(
                        nn.Linear(self.chart_features, d_model//2),
                        nn.ReLU(), 
                        nn.Linear(d_model//2, d_model)
                )
                self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
                self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_encoders) 
                self.attentive_pooling = AttentivePooling(input_dim=self.d_model) 
                self.fc = nn.Linear(self.d_model, self.num_classes) 
                self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc) 
        def forward(self, x): 
                x = self.batchnorm(x) 
                x = self.chart_embedder(x) 
                x = self.pos_encoder(x) 
                x = self.transformer_encoder(x) 
                x = self.attentive_pooling(x) 
                x = self.multi_dropout(x) 
                return x 
            
### get data and preprocess ### 
with open("BTC_USDT-4h-12.json") as f: 
        d = json.load(f) 

chart_df = pd.DataFrame(d) 
chart_df = chart_df.rename(columns={0:"timestamp", 1:"open", 2:"high", 3:"low", 4:"close", 5:"volume"})

def process(df): 
        binance = ccxt.binance() 
        dates = df["timestamp"].values 
        timestamp = [] 
        for i in range(len(dates)):
                date_string = binance.iso8601(int(dates[i])) 
                date_string = date_string[:10] + " " + date_string[11:-5] 
                timestamp.append(date_string) 
        df["datetime"] = timestamp
        df = df.drop(columns={"timestamp"}) 
        return df 

chart_df = process(chart_df) 

hours, days, months, years = [],[],[],[] 
for dt in tqdm(chart_df["datetime"]):
        dtobj = pd.to_datetime(dt) 
        hour = dtobj.hour 
        day = dtobj.day 
        month = dtobj.month 
        year = dtobj.year 
        hours.append(hour) 
        days.append(day) 
        months.append(month) 
        years.append(year) 

chart_df["hours"] = hour 
chart_df["days"] = day 
chart_df["months"] = months 
chart_df["years"] = years 

def preprocess_seq_data(chart_df, threshold=0.005):
        targets = [] 
        openv = chart_df["open"].values 
        close = chart_df["close"].values 
        high = chart_df["high"].values 
        low = chart_df["low"].values 
        volume = chart_df["volume"].values  
        for i in range(close.shape[0] - 1):
            high_vol = (high[i+1] - close[i]) / close[i] 
            low_vol = (low[i+1] - close[i]) / close[i] 
            if high_vol >= threshold: 
                targets.append(1) 
            else:
                targets.append(0) 
            '''
            if high_vol >= threshold:
                targets.append(0) 
            elif low_vol <= -threshold:
                targets.append(1) 
            else:
                targets.append(2)  
            ''' 
        targets.append(None) 
        chart_df["targets"] = targets 
        chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True) 
        chart_df["bop"] = chart_df.ta.bop(lookahead=False) 
        chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False) 
        chart_df["cmf"] = chart_df.ta.cmf(lookahead=False) 
        chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100 
        chart_df["high/low"] = chart_df["high"] / chart_df["low"] 
        chart_df["high/open"] = chart_df["high"] / chart_df["open"] 
        chart_df["low/open"] = chart_df["low"] / chart_df["open"] 
        chart_df["close/open"] = chart_df["close"] / chart_df["open"] 
        chart_df["high/close"] = chart_df["high"] / chart_df["close"] 
        chart_df["low/close"] = chart_df["low"] / chart_df["close"] 
        ratio_open, ratio_high, ratio_low, ratio_close, ratio_volume = [None], [None], [None], [None], [None] 
        for i in range(1, len(openv)):
                r_open = openv[i] / openv[i-1] 
                r_high = high[i] / high[i-1] 
                r_low = low[i] / low[i-1]  
                r_close = close[i] / close[i-1] 
                if volume[i-1] == 0:
                        r_vol = 1 
                else:
                        r_vol = volume[i] / volume[i-1] 
                ratio_open.append(r_open) 
                ratio_close.append(r_close)  
                ratio_high.append(r_high) 
                ratio_low.append(r_low) 
                ratio_volume.append(r_vol) 
        chart_df["r_open"] = ratio_open 
        chart_df["r_close"] = ratio_close 
        chart_df["r_high"] = ratio_high 
        chart_df["r_low"] = ratio_low 
        chart_df["r_volume"] = ratio_volume 
        chart_df.dropna(inplace=True) 
        return chart_df 

chart_df = preprocess_seq_data(chart_df) 

X = chart_df[["bop", "ebsw", "cmf", "rsi/100", "r_open", "r_close", "r_high", "r_low", "r_volume", "high/low", "high/open", "low/open", "close/open", "high/close", "low/close"]]
Y = chart_df["targets"]  

train_size = int(0.8 * X.shape[0]) 
val_size = int(0.1 * X.shape[0]) 

X_train = X[:train_size]
Y_train = Y[:train_size] 

X_val = X[train_size:train_size+val_size] 
Y_val = Y[train_size:train_size+val_size] 

X_test = X[train_size+val_size:] 
Y_test = Y[train_size+val_size:] 

class_weights = compute_class_weight(class_weight = "balanced",
                                     classes = np.unique(Y_train),
                                     y = Y_train) 
d = {0:class_weights[0], 1:class_weights[1]} 

clf = XGBClassifier(silent=False, 
                    n_estimators=200,
                    class_weight=d, 
                    metric="logloss")

clf.fit(X_train, 
        Y_train, 
        eval_set=[(X_val, Y_val)],
        verbose=20)


Y_pred = clf.predict(X_test) 

cnt = 0 
for i in range(len(Y_pred)):
    if Y_pred[i] == Y_test[i]: 
        cnt += 1 
print(f"accuracy = {cnt/len(Y_pred) * 100.0}%") 
f1 = f1_score(Y_test, Y_pred, average="macro")
print(f"f1 score = {f1}") 

joblib.dump(clf, "xgb_high") 
print("done saving!") 


