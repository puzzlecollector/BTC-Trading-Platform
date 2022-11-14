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

''' implement loss function ''' 
class WeightedFocalLoss(nn.Module): 
    def __init__(self, alpha, gamma=2): 
        super(WeightedFocalLoss, self).__init__() 
        self.alpha = alpha 
        self.device = torch.device("cuda") 
        self.alpha = self.alpha.to(self.device) 
        self.gamma = gamma 
    def forward(self, inputs, targets): 
        CE_loss = nn.CrossEntropyLoss()(inputs, targets) 
        targets = targets.type(torch.long) 
        at = self.alpha.gather(0, targets.data.view(-1)) 
        pt = torch.exp(-CE_loss) 
        F_loss = at * (1-pt)**self.gamma * CE_loss 
        return F_loss.mean() 
    
def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return np.sum(pred_flat==labels_flat) / len(labels_flat) 

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
                output = self.LM(input_id, attention_mask)[0][:,0] 
                news_v[i,j,:] = output 
        news_v = self.news_pos_encoder(news_v) 
        news_v = self.news_transformer_encoder(news_v) 
        news_v = self.news_attentive_pooling(news_v) 
        embedding = torch.cat((chart_v, news_v), dim=1) 
        x = self.multi_dropout(embedding) 
        return x 

## load data ## 
with open("X_seq.pkl", "rb") as f: 
    X_seq = pickle.load(f) 
    
with open("Y_labels.pkl", "rb") as f: 
    Y_labels_arr = pickle.load(f) 
    
full_input_ids = torch.load("full_input_ids.pt") 
full_attn_masks = torch.load("full_attention_masks.pt") 
    
X_seq = torch.tensor(X_seq).float() 
Y_labels = torch.tensor(Y_labels_arr, dtype=int) 

print(X_seq.shape, Y_labels.shape, full_input_ids.shape, full_attn_masks.shape) 

train_size = int(0.8 * X_seq.shape[0]) 
val_size = int(0.1 * X_seq.shape[0]) 

# compute class weights 
train_Y_labels_arr = Y_labels_arr[:train_size] 
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_Y_labels_arr), y=np.array(train_Y_labels_arr)) 
print(f"class weights : {class_weights}") 

train_seq = X_seq[:train_size] 
train_labels = Y_labels[:train_size] 
train_input_ids = full_input_ids[:train_size] 
train_attn_masks = full_attn_masks[:train_size] 

val_seq = X_seq[train_size:train_size+val_size] 
val_labels = Y_labels[train_size:train_size+val_size] 
val_input_ids = full_input_ids[train_size:train_size+val_size] 
val_attn_masks = full_attn_masks[train_size:train_size+val_size] 

test_seq = X_seq[train_size+val_size:] 
test_labels = Y_labels[train_size+val_size:] 
test_input_ids = full_input_ids[train_size+val_size:] 
test_attn_masks = full_attn_masks[train_size+val_size:] 

batch_size = 6
train_data = TensorDataset(train_seq, train_input_ids, train_attn_masks, train_labels) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 

val_data = TensorDataset(val_seq, val_input_ids, val_attn_masks, val_labels) 
val_sampler = SequentialSampler(val_data) 
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size) 

test_data = TensorDataset(test_seq, test_input_ids, test_attn_masks, test_labels) 
test_sampler = SequentialSampler(test_data) 
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size) 

print(train_seq.shape, val_seq.shape, test_seq.shape) 

## define model ## 
train_losses, val_losses = [], [] 
train_accuracies, val_accuracies = [], [] 
device = torch.device("cuda") 
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
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
epochs = 20 
total_steps = len(train_dataloader) * epochs  
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps) 
class_weights = torch.tensor(class_weights, dtype=torch.float) 
loss_func = WeightedFocalLoss(alpha=class_weights) 
model.zero_grad() 
for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs): 
    train_loss, train_accuracy = 0, 0 
    model.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch: 
        for step, batch in enumerate(tepoch): 
            batch = tuple(t.to(device) for t in batch) 
            b_seq_data, b_input_ids, b_attn_masks, b_labels = batch 
            outputs = model(b_seq_data, b_input_ids, b_attn_masks) 
            loss = loss_func(outputs, b_labels) 
            train_loss += loss.item() 
            logits_cpu, labels_cpu = outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy() 
            train_accuracy += flat_accuracy(logits_cpu, labels_cpu) 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
            tepoch.set_postfix(loss=train_loss / (step+1), accuracy=100.0 * train_accuracy / (step+1)) 
            time.sleep(0.1) 
        avg_train_loss = train_loss / len(train_dataloader) 
        avg_train_accuracy = train_accuracy / len(train_dataloader) 
        train_losses.apend(avg_train_loss) 
        train_accuracies.append(avg_train_accuracy) 
        print(f"average train loss : {avg_train_loss}") 
        print(f"average train accuracy : {avg_train_accuracy}") 
    val_loss, val_accuracy = 0, 0
    model.eval() 
    for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)): 
        batch = tuple(t.to(device) for t in batch) 
        b_seq_data, b_input_ids, b_attn_masks, b_labels = batch
        with torch.no_grad(): 
            outputs = model(b_seq_data, b_input_ids, b_attn_masks) 
        loss = loss_func(outputs, b_labels) 
        val_loss += loss.item() 
        logits_cpu, labels_cpu = outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy() 
        val_accuracy += flat_accuracy(logits_cpu, labels_cpu) 
    avg_val_loss = val_loss / len(val_dataloader) 
    avg_val_accuracy = val_accuracy / len(val_dataloader) 
    val_losses.append(avg_val_loss) 
    val_accuracies.append(avg_val_accuracy) 
    print(f"average val loss : {avg_val_loss}") 
    print(f"average val accuracy : {avg_val_accuracy}") 
    print("saving current checkpoint...") 
    if val_accuracies[-1] == np.max(val_accuracies): 
        torch.save(model.state_dict(), f"TFNet_MultiModal_epochs:{epoch_i+1}_val_acc:{avg_val_accuracy}_val_loss:{avg_val_loss}.pt")  
