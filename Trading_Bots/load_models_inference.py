import pandas as pd 
import numpy as np 
import seaborn as sns 
from tqdm.auto import tqdm 
from sklearn import preprocessing
import time 
import datetime
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 
from transformers import *
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pandas_ta as ta

#########################
### get XGBoost model ###
#########################
xgb_clf = XGBClassifier() 
xgb_clf.load_model("CBITS_XGB") 

#####################
### get DNN model ###
#####################
class MultiSampleDropout(nn.Module):
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples
    def forward(self, out): 
        return torch.mean(torch.stack([self.classifier(self.dropout(p=rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)

class AttentivePooling(torch.nn.Module):
    def __init__(self, input_dim):
        super(AttentivePooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
    def forward(self, x):
        softmax = F.softmax
        att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1)
        x = torch.sum(x * att_w, dim=1)
        return x
    
class DNN(nn.Module): 
    def __init__(self, num_classes, num_features): 
        super(DNN, self).__init__() 
        self.num_classes = num_classes 
        self.num_features = num_features
        self.batchnorm = nn.BatchNorm1d(self.num_features) 
        self.fc = nn.Linear(self.num_features, 128) 
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, self.num_classes) 
        self.multi_dropout = MultiSampleDropout(0.2, 4, self.fc3)
    def forward(self, x):
        x = self.batchnorm(x) 
        x = self.fc(x) 
        x = self.fc2(x) 
        x = self.multi_dropout(x) 
        return x 

device = torch.device("cuda") # running on GPU server 
model = DNN(num_classes=3, num_features=81)
chkpt = torch.load("DNN_chkpt.pt") 
model.load_state_dict(chkpt) 
model.to(device) 
model.eval() 

################################
### preprocess data function ###
################################
def preprocess_dataframe(chart_df): 
    months, days, hours = [], [], [] 
    datetime_values = chart_df["datetime"].values 
    for i in range(len(datetime_values)): 
        dtobj = pd.to_datetime(datetime_values[i]) 
        months.append(dtobj.month) 
        days.append(dtobj.day) 
        hours.append(dtobj.hour) 
    chart_df["months"] = months 
    chart_df["days"] = days 
    chart_df["hours"] = hours
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)
    chart_df["bop"] = chart_df.ta.bop(lookahead=False) 
    chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False) 
    chart_df["cmf"] = chart_df.ta.cmf(lookahead=False) 
    chart_df["vwap"] = chart_df.ta.vwap(lookahead=False) 
    chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100 
    chart_df["high/low"] = chart_df["high"] / chart_df["low"] 
    chart_df["close/open"] = chart_df["close"] / chart_df["open"] 
    chart_df["high/open"] = chart_df["high"] / chart_df["open"] 
    chart_df["low/open"] = chart_df["low"] / chart_df["open"] 
    chart_df["hwma"] = chart_df.ta.hwma(lookahead=False) 
    chart_df["linreg"] = chart_df.ta.linreg(lookahead=False) 
    chart_df["hwma/close"] = chart_df["hwma"] / chart_df["close"] 
    chart_df["linreg/close"] = chart_df["linreg"] / chart_df["close"] 
    for l in tqdm(range(1, 12), position = 0, leave=True): 
        for col in ["open", "high", "low", "close", "volume", "vwap"]: 
            val = chart_df[col].values 
            val_ret = [None for _  in range(l)] 
            for i in range(l, len(val)):
                if val[i-l] == 0: 
                    ret = 1 
                else: 
                    ret = val[i] / val[i-l]  
                val_ret.append(ret) 
            chart_df["{}_change_{}".format(col, l)] = val_ret 

    chart_df.drop(columns={"open", "high", "low", "close", "volume", "vwap", "hwma", "linreg", "datetime"}, inplace=True) 
    chart_df.dropna(inplace=True) 
    return chart_df

def news_scraping_module(): 
    pass 
