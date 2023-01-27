'''
!pip install transformers 
!pip install pandas-ta 
!pip install imblearn 
!pip install seaborn 
!pip install ccxt 
!pip install sentencepiece
''' 
import numpy as np 
import pandas as pd 
import json 
import ccxt 
import seaborn as sns
import os 
import pandas_ta as ta 
import time
from datetime import datetime, timedelta
import math
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt 
from transformers import * 
import torch 
from torch import Tensor 
from torch.utils.data import * 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler

with open("BTC_USDT-4h-12.json") as f: 
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

# create targets 
# 1% up: 0 
# 1% down: 1 
# less than 1%: 2  
close = chart_df["close"].values 

targets = [] 

for i in tqdm(range(close.shape[0]-1)): 
    ret = (close[i+1] - close[i]) / close[i] * 100 
    if ret >= 0.8: 
        targets.append(0) 
    elif ret <= -0.8:
        targets.append(1) 
    else:
        targets.append(2) 
targets.append(None) 

chart_df["targets"] = targets 

chart_df.dropna(inplace=True)

# preprocess data  
chart_df["cmf"] = chart_df.ta.cmf(lookahead=False) 
chart_df["rsi"] = chart_df.ta.rsi(lookahead=False) / 100 

open_r, high_r, low_r, close_r, volume_r = [None], [None], [None], [None], [None] 
openv = chart_df["open"].values 
highv = chart_df["high"].values  
lowv = chart_df["low"].values 
closev = chart_df["close"].values 
volumev = chart_df["volume"].values 

for i in range(1, len(openv)): 
    r_open = openv[i] / openv[i-1] 
    r_high = highv[i] / openv[i-1] 
    r_low = lowv[i] / openv[i-1] 
    r_close = closev[i] / openv[i-1] 
    if volumev[i-1] != 0: 
        r_vol = volumev[i] / volumev[i-1]
    else:
        r_vol = 1 # error in data maybe we can interpolate? 
    open_r.append(r_open) 
    high_r.append(r_high) 
    low_r.append(r_low) 
    close_r.append(r_close) 
    volume_r.append(r_vol) 

chart_df["open_ratio"] = open_r 
chart_df["high_ratio"] = high_r 
chart_df["low_ratio"] = low_r 
chart_df["close_ratio"] = close_r 
chart_df["volume_ratio"] = volume_r 

chart_df.dropna(inplace=True) 

chart_df.drop(columns={"open", "high", "low", "close", "volume"}, inplace=True) 


rus = RandomUnderSampler(random_state=64) # switch random seeds to create multiple traders 
X = chart_df[["datetime", "cmf", "rsi", "open_ratio", "high_ratio", "low_ratio", "close_ratio", "volume_ratio"]] 
Y = chart_df["targets"] 
X, Y = rus.fit_resample(X, Y)
chart_df = pd.concat([X, Y], axis=1) 
chart_df = chart_df.sort_values(by=["datetime"])

datetimes = chart_df["datetime"].values 
datetimes_dt = [] 

for i in tqdm(range(len(datetimes)), position=0, leave=True): 
    dt = datetime.strptime(str(datetimes[i]), "%Y-%m-%d %H:%M:%S") 
    datetimes_dt.append(dt) 

news = pd.read_csv("full_news_22_01_16.csv")
news_dt = [] 

years, months, days, hours = news["year"].values, news["month"].values, news["day"].values, news["hour"].values 

for i in tqdm(range(len(years)), position=0, leave=True):
    datestr = str(years[i]) + "-" + str(months[i]) + "-" + str(days[i]) + " " + str(hours[i]) 
    dt = datetime.strptime(datestr, "%Y-%m-%d %H") 
    news_dt.append(dt) 

titles = news["titles"].values 
contents = news["contents"].values 

corresponding_headlines = [] 

for i in tqdm(range(len(datetimes_dt)), position=0, leave=True):
    start, end = datetimes_dt[i], datetimes_dt[i] + timedelta(hours=4) 
    cur_titles = [] 
    for j in range(len(news_dt)): 
        if news_dt[j] > end: 
            break 
        if news_dt[j] >= start and news_dt[j] <= end: 
            cur_titles.append(titles[j]) 
    concatenated_titles = "" 
    for j in range(len(cur_titles)): 
        if j < len(cur_titles)-1: 
            concatenated_titles += cur_titles[j] + "[SEP]" 
        else:
            concatenated_titles += cur_titles[j] 
    corresponding_headlines.append(concatenated_titles) 
                
        
chart_df = chart_df.drop(columns={"datetime"}) 

seq_len = 1

X_chart, X_news = [], []  
Y = [] 

for i in tqdm(range(chart_df.shape[0] - seq_len), position=0, leave=True): 
    cur_seq = chart_df.iloc[i:i+seq_len, 1:].values  
    cur_target = chart_df.iloc[i+seq_len-1, 0] 
    X_chart.append(cur_seq) 
    Y.append(cur_target) 
    
    range_news = "" 
    for j in range(seq_len): 
        if len(corresponding_headlines[i+j]) > 0: 
            if j < seq_len-1: 
                range_news += corresponding_headlines[i+j] + "[SEP]" 
            else: 
                range_news += corresponding_headlines[i+j] 
    X_news.append(range_news) 
    
X_chart = torch.tensor(X_chart).float()  
Y = torch.tensor(Y, dtype=int) 

tokenizer = AutoTokenizer.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 

input_ids, attn_masks = [], [] 

for i in tqdm(range(len(X_news))): 
    encoded_inputs = tokenizer(X_news[i], max_length=512, truncation=True, padding="max_length") 
    input_ids.append(encoded_inputs["input_ids"]) 
    attn_masks.append(encoded_inputs["attention_mask"])  

input_ids = torch.tensor(input_ids, dtype=int) 
attn_masks = torch.tensor(attn_masks, dtype=int) 

print(input_ids.shape, attn_masks.shape) 

train_size = int(0.8 * X_chart.shape[0]) 
val_size = int(0.1 * X_chart.shape[0]) 
test_size = int(0.1 * X_chart.shape[0]) 

train_chart, train_input_ids, train_attn_masks, train_targets = X_chart[:train_size], input_ids[:train_size], attn_masks[:train_size], Y[:train_size] 
val_chart, val_input_ids, val_attn_masks, val_targets = X_chart[train_size:train_size+val_size], input_ids[train_size:train_size+val_size], attn_masks[train_size:train_size+val_size], Y[train_size:train_size+val_size] 
test_chart, test_input_ids, test_attn_masks, test_targets = X_chart[train_size+val_size:], input_ids[train_size+val_size:], attn_masks[train_size+val_size:], Y[train_size+val_size:] 

print(train_chart.shape, val_chart.shape, test_chart.shape, train_input_ids.shape, val_input_ids.shape, test_input_ids.shape, train_targets.shape, val_targets.shape, test_targets.shape) 

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
        self.register_buffer("pe", pe) 
    def forward(self, x): 
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x) 

# https://arxiv.org/abs/1905.09788
class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout 
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples 
    def forward(self, out): 
        return torch.mean(torch.stack([self.classifier(self.dropout(p=rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)

# there are potentially better pooling methods 
class AttentivePooling(torch.nn.Module): 
    def __init__(self, input_dim): 
        super(AttentivePooling, self).__init__() 
        self.W = nn.Linear(input_dim, 1) 
    def forward(self, x): 
        softmax = F.softmax 
        att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1) 
        x = torch.sum(x * att_w, dim=1) 
        return x 
    
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_masks):
        input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings    

class BitTrader(nn.Module): 
    def __init__(self, chart_features, sequence_length, d_model, num_classes, n_heads, num_encoders, plm, multi_dropout_r, multi_dropout_samples, chart_reduced_dim, news_reduced_dim): 
        super(BitTrader, self).__init__() 
        # intialize necessary variables  
        self.chart_features = chart_features 
        self.sequence_length = sequence_length  
        self.d_model = d_model 
        self.num_classes = num_classes  
        self.n_heads = n_heads 
        self.num_encoders = num_encoders  
        self.plm = plm 
        self.multi_dropout_r = multi_dropout_r 
        self.multi_dropout_samples = multi_dropout_samples 
        self.chart_reduced_dim = chart_reduced_dim 
        self.news_reduced_dim = news_reduced_dim
        
        # chart encoder 
        self.chart_embedder = nn.Sequential(
            nn.Linear(self.chart_features, d_model//2), 
            nn.ReLU(), 
            nn.Linear(d_model//2, d_model) 
        ) 
        self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_encoders) 
        self.attentive_pooling = AttentivePooling(input_dim=self.d_model)    
        
        # news encoder 
        self.lm = AutoModel.from_pretrained(self.plm) 
        self.mean_pooler = MeanPooling() 
        self.lm_config = AutoConfig.from_pretrained(self.plm) 
        
        # chart fc-layer 
        self.chart_fc = nn.Linear(self.d_model, self.chart_reduced_dim) # 256 -> 128 
        self.chart_multi_dropout = MultiSampleDropout(0.2, 8, self.chart_fc) 
        
        # news fc-layer 
        self.news_fc = nn.Linear(self.lm_config.hidden_size, self.news_reduced_dim) # 768 -> 128 
        self._init_weights(self.news_fc) 
        self.news_multi_dropout = MultiSampleDropout(0.2, 8, self.news_fc) 
        
        # final output 
        self.output_fc = nn.Linear(self.chart_reduced_dim + self.news_reduced_dim, self.num_classes) 
        self.output_multi_dropout = MultiSampleDropout(0.2, 8, self.output_fc) 
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.lm_config.initializer_range) 
            if module.bias is not None: 
                module.bias.data.zero_() 
        
    def forward(self, x_chart, input_ids, attn_masks): 
        # encode chart data 
        x_chart = self.chart_embedder(x_chart)
        x_chart = self.pos_encoder(x_chart) 
        x_chart = self.transformer_encoder(x_chart) 
        x_chart = self.attentive_pooling(x_chart) 
        x_chart = self.chart_multi_dropout(x_chart)  
        
        # encode news data  
        x_news = self.lm(input_ids, attn_masks)[0] 
        x_news = self.mean_pooler(x_news, attn_masks) 
        x_news = self.news_multi_dropout(x_news) 
        
        # calculate output logit 
        x = torch.cat((x_chart, x_news), dim=1)  
        x = self.output_multi_dropout(x) 
        return x 
        
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat) 

# define dataloaders 
batch_size = 16
train_data = TensorDataset(train_chart, train_input_ids, train_attn_masks, train_targets) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 

val_data = TensorDataset(val_chart, val_input_ids, val_attn_masks, val_targets) 
val_sampler = SequentialSampler(val_data) 
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

train_losses, train_accuracies, train_f1s = [], [], []   
val_losses, val_accuracies, val_f1s = [], [], [] 

best_loss = 99999999999999999 
best_f1 = 0 
best_accuracy = 0 
# class_weights = class_weights.to(device).float()
# loss_func = nn.CrossEntropyLoss(weight = class_weights) # weighted CE Loss 

loss_func = nn.CrossEntropyLoss() 

model = BitTrader(chart_features=train_chart.shape[2],
                  sequence_length=train_chart.shape[1],
                  d_model=256,
                  num_classes=3,
                  n_heads=4,
                  num_encoders=3, 
                  plm="totoro4007/cryptodeberta-base-all-finetuned",
                  multi_dropout_r=0.2,
                  multi_dropout_samples=8, 
                  chart_reduced_dim=128, 
                  news_reduced_dim=128)

model.to(device) 
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
epochs = 20 
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps) 
model.zero_grad() 
for epoch_i in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs): 
    train_loss, train_accuracy = 0, 0 
    model.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch: 
        for step, batch in enumerate(tepoch): 
            batch = tuple(t.to(device) for t in batch) 
            b_chart, b_input_ids, b_attn_masks, b_labels = batch 
            outputs = model(b_chart, b_input_ids, b_attn_masks) 
            # get loss 
            loss = loss_func(outputs, b_labels) 
            train_loss += loss.item() 
            
            # get accuracy 
            pred_logits, gt_logits = outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy() 
            train_accuracy += flat_accuracy(pred_logits, gt_logits) 
            
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
            tepoch.set_postfix(loss=train_loss/(step+1), accuracy=train_accuracy/(step+1)) 
            time.sleep(0.1) 
    avg_train_loss = train_loss / len(train_dataloader) 
    avg_train_accuracy = train_accuracy / len(train_dataloader) 
    print(f"train loss : {avg_train_loss} | train accuracy : {avg_train_accuracy}")  
    
    val_loss, val_accuracy = 0, 0 
    model.eval() 
    for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)):
        batch = tuple(t.to(device) for t in batch) 
        b_chart, b_input_ids, b_attn_masks, b_labels = batch 
        with torch.no_grad(): 
            outputs = model(b_chart, b_input_ids, b_attn_masks) 
        # get loss 
        loss = loss_func(outputs, b_labels) 
        val_loss += loss.item() 
        
        # get accuracy 
        pred_logits, gt_logits = outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy() 
        val_accuracy += flat_accuracy(pred_logits, gt_logits) 
    avg_val_loss = val_loss / len(val_dataloader) 
    avg_val_accuracy = val_accuracy / len(val_dataloader) 
    print(f"validation loss : {avg_val_loss} | validation accuracy : {avg_val_accuracy}") 
    
    if best_loss > avg_val_loss: 
        best_loss = avg_val_loss 
        torch.save(model.state_dict(), "Best_checkpoint.pt") 
        


