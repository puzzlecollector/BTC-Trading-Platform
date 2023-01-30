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

chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)

chart_df["cmf"] = chart_df.ta.cmf(lookahead=False) 
chart_df["rsi"] = chart_df.ta.rsi(lookahead=False) / 100 
chart_df["bop"] = chart_df.ta.bop(lookahead=False) 
chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False) 
chart_df["vwap"] = chart_df.ta.vwap(lookahead=False) 

chart_df.dropna(inplace=True)

rus_seed = 42

rus = RandomUnderSampler(random_state=rus_seed) # switch random seeds to create multiple traders 

input_columns = [] 
for col in chart_df.columns:
    if col != "targets": 
        input_columns.append(col) 

X = chart_df[input_columns] 
Y = chart_df["targets"] 

XU, YU = rus.fit_resample(X.iloc[41:], Y.iloc[41:])
UChart_df = pd.concat([XU, YU], axis=1) 
UChart_df = UChart_df.sort_values(by=["datetime"])

seq_len = 42 

date_chart_dict = {} 

datetimes = chart_df["datetime"].values 

for i in tqdm(range(len(datetimes)), position=0, leave=True): 
    if i+1-seq_len < 0: 
        continue 
    date_chart_dict[datetimes[i]] = chart_df.iloc[i+1-seq_len:i+1, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]].values

    
Udatetimes = UChart_df["datetime"].values 
targets = UChart_df["targets"].values 

X_chart, Y_targets = [], [] 

for i in tqdm(range(len(Udatetimes)), position=0, leave=True):
    X_chart.append(date_chart_dict[Udatetimes[i]]) 
    Y_targets.append(targets[i]) 
    
X_chart = torch.tensor(X_chart).float() 
Y_targets = torch.tensor(Y_targets, dtype=int) 


train_size = int(X_chart.shape[0] * 0.8) 
val_size = int(X_chart.shape[0] * 0.1) 

X_train, Y_train = X_chart[:train_size], Y_targets[:train_size] 
X_val, Y_val = X_chart[train_size:train_size+val_size], Y_targets[train_size:train_size+val_size] 
X_test, Y_test = X_chart[train_size+val_size:], Y_targets[train_size+val_size:] 

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape) 

batch_size = 32
train_data = TensorDataset(X_train, Y_train) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 

val_data = TensorDataset(X_val, Y_val) 
val_sampler = SequentialSampler(val_data) 
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

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

class DAIN_Layer(nn.Module):
    def __init__(self, mode, mean_lr, gate_lr, scale_lr, input_dim):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std

            # Step 3: 
            avg = torch.mean(x, 2)
            gate = F.sigmoid(self.gating_layer(avg))
            gate = gate.resize(gate.size(0), gate.size(1), 1)
            x = x * gate

        else:
            assert False
        return x
    
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
        
        # chart normalization layer 
        self.dain = DAIN_Layer(mode="adaptive_avg", mean_lr=1e-06, gate_lr=10, scale_lr=0.001, input_dim=self.sequence_length)  
        
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
        self.chart_multi_dropout = MultiSampleDropout(self.multi_dropout_r, self.multi_dropout_samples, self.chart_fc) 
        
        # news fc-layer 
        self.news_fc = nn.Linear(self.lm_config.hidden_size, self.news_reduced_dim) # 768 -> 128 
        self._init_weights(self.news_fc) 
        self.news_multi_dropout = MultiSampleDropout(self.multi_dropout_r, self.multi_dropout_samples, self.news_fc) 
        
        # final output 
        # self.output_fc = nn.Linear(self.chart_reduced_dim + self.news_reduced_dim, self.num_classes) 
        self.output_fc = nn.Linear(self.chart_reduced_dim, self.num_classes) 
        self.output_multi_dropout = MultiSampleDropout(self.multi_dropout_r, self.multi_dropout_samples, self.output_fc) 
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.lm_config.initializer_range) 
            if module.bias is not None: 
                module.bias.data.zero_() 
        
    def forward(self, x_chart): #, input_ids, attn_masks): 
        # encode chart data 
        x_chart = self.dain(x_chart) 
        x_chart = self.chart_embedder(x_chart) 
        x_chart = self.pos_encoder(x_chart) 
        x_chart = self.transformer_encoder(x_chart) 
        x_chart = self.attentive_pooling(x_chart) 
        x_chart = self.chart_multi_dropout(x_chart)  
        x = self.output_multi_dropout(x_chart) 
        return x
        '''
        # encode news data  
        x_news = self.lm(input_ids, attn_masks)[0] 
        x_news = self.mean_pooler(x_news, attn_masks) 
        x_news = self.news_multi_dropout(x_news) 
        
        # calculate output logit 
        x = torch.cat((x_chart, x_news), dim=1)  
        x = self.output_multi_dropout(x) 
        return x 
        ''' 

model = BitTrader(chart_features=10,
                  sequence_length=42,
                  d_model=256,
                  num_classes=3,
                  n_heads=8,
                  num_encoders=6,
                  plm="totoro4007/cryptodeberta-base-all-finetuned",
                  multi_dropout_r=0.2,
                  multi_dropout_samples=8,
                  chart_reduced_dim=128,
                  news_reduced_dim=128)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat) 

loss_func = nn.CrossEntropyLoss() 
best_loss = 9999999999
model.to(device) 
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
epochs = 1000
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps) 
model.zero_grad() 
for epoch_i in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs): 
    train_loss, train_accuracy = 0, 0 
    model.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch: 
        for step, batch in enumerate(tepoch): 
            batch = tuple(t.to(device) for t in batch) 
            # b_chart, b_input_ids, b_attn_masks, b_labels = batch
            # outputs = model(b_chart, b_input_ids, b_attn_masks) 
            b_chart, b_labels = batch 
            outputs = model(b_chart) 
            
            
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
        # b_chart, b_input_ids, b_attn_masks, b_labels = batch 
        b_chart, b_labels = batch 
        
        with torch.no_grad(): 
            # outputs = model(b_chart, b_input_ids, b_attn_masks)  
            outputs = model(b_chart) 
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
        torch.save(model.state_dict(), f"{rus_seed}_Best_checkpoint.pt") 
        

os.rename(f"{rus_seed}_Best_checkpoint.pt", f"{rus_seed}_Best_checkpoint_{best_loss}.pt")
