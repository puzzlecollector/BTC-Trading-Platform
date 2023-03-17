import pandas as pd 
import numpy as np 
import json 
import ccxt 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pickle
from torch.nn import Transformer 
from torch import nn 
import torch 
import math 
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler, IterableDataset 

with open("BTC_USDT-4h-14.json") as f: 
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

hours, days, months, years = [], [], [], [] 
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

chart_df["hours"] = hours
chart_df["days"] = days 
chart_df["months"] = months 
chart_df["years"] = years 

data = chart_df[["open", "high", "low", "close", "volume"]].values 

input_window, forecast_window = 24, 6

X, Y = [], [] 

for i in tqdm(range(data.shape[0] - input_window - forecast_window), position=0, leave=True): 
    X.append(data[i:i+input_window,:]) 
    Y.append(data[i+input_window:i+input_window+forecast_window, 3]) 

X = torch.tensor(X).float() 
Y = torch.tensor(Y).float() 

train_size = int(X.shape[0] * 0.8) 
val_size = int(X.shape[0] * 0.1) 

X_train = X[:train_size]
Y_train = Y[:train_size] 

X_val = X[train_size:train_size+val_size] 
Y_val = Y[train_size:train_size+val_size] 

X_test = X[train_size+val_size:] 
Y_test = Y[train_size+val_size:] 

device = torch.device("cuda:0") 
target_n = 1 # 맞춰야하는 피처의 수 
learning_rate = 5e-4 
batch_size = 128 
EPOCHS = 1000
teacher_forcing = False 
n_layers = 3 # rnn 레이어 층 
dropout = 0.2 
window_size = 24 # 인코더 시퀀스 길이  
future_size = 12 # 디코더 시퀀스 길이 
hidden_dim = 128 # rnn 히든차원 
save_path = "best_seq2seq.pt" 

class CustomDataset(Dataset): 
    def __init__(self, encoder_input, decoder_input): 
        self.encoder_input = encoder_input 
        self.decoder_input = decoder_input 
    
    def __len__(self): 
        return len(self.encoder_input) 
    
    def __getitem__(self, i): 
        return {
            "encoder_input": torch.tensor(self.encoder_input[i], dtype=torch.float32), 
            "decoder_input": torch.tensor(self.decoder_input[i], dtype=torch.float32) 
        } 
    
train_dataset = CustomDataset(X_train, Y_train) 
val_dataset = CustomDataset(X_val, Y_val) 

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

class Encoder(nn.Module): 
    def __init__(self, input_dim, hidden_dim, n_layers, dropout): 
        super().__init__() 
        self.n_layers = n_layers 
        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self, inp_seq): 
        inp_seq = inp_seq.permute(1,0,2) 
        outputs, hidden = self.rnn(inp_seq) 
        return outputs, hidden
    
class BahdanauAttention(nn.Module): 
    def __init__(self, dec_output_dim, units): 
        super(BahdanauAttention, self).__init__() 
        self.W1 = nn.Linear(dec_output_dim, units) 
        self.W2 = nn.Linear(dec_output_dim, units) 
        self.V = nn.Linear(dec_output_dim, 1) 
    
    def forward(self, hidden, enc_output): 
        query_with_time_axis = hidden.unsqueeze(1) 
        score = self.V(torch.tanh(self.W1(query_with_time_axis) + self.W2(enc_output))) 
        attention_weights = torch.softmax(score, axis=1) 
        context_vector = attention_weights * enc_output 
        context_vector = torch.sum(context_vector, dim = 1) 
        return context_vector, attention_weights 
    
class Decoder(nn.Module): 
    def __init__(self, dec_feature_size, encoder_hidden_dim, output_dim, decoder_hidden_dim, n_layers, dropout, attention):
        super().__init__() 
        self.output_dim = output_dim 
        self.decoder_hidden_dim = decoder_hidden_dim 
        self.n_layers = n_layers 
        self.attention = attention 
        self.layer = nn.Linear(dec_feature_size, encoder_hidden_dim) 
        self.rnn = nn.GRU(encoder_hidden_dim*2, decoder_hidden_dim, n_layers, dropout=dropout) 
        self.fc_out = nn.Linear(hidden_dim, output_dim) 
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, enc_output, dec_input, hidden): 
        dec_input = torch.reshape(dec_input, (-1, 1)) 
        dec_input = self.layer(dec_input) 
        context_vector, attention_weight = self.attention(hidden, enc_output) 
        dec_input = torch.cat([torch.sum(context_vector, dim=0), dec_input], dim=1) 
        dec_input = dec_input.unsqueeze(0) 
        output, hidden = self.rnn(dec_input, hidden) 
        prediction = self.fc_out(output.sum(0)) 
        return prediction, hidden
    
class Seq2Seq(nn.Module): 
    def __init__(self, encoder, decoder, attention): 
        super().__init__() 
        self.encoder = encoder
        self.decoder = decoder 
    
    def forward(self, encoder_input, decoder_input, teacher_forcing=False):
        batch_size = decoder_input.size(0)
        trg_len = decoder_input.size(1) 
        outputs = torch.zeros(batch_size, trg_len-1, self.decoder.output_dim).to(device) 
        enc_output, hidden = self.encoder(encoder_input) 
        dec_input = decoder_input[:,0] 
        for t in range(1, trg_len): 
            output, hidden = self.decoder(enc_output, dec_input, hidden) 
            outputs[:,t-1] = output
            if teacher_forcing == True: 
                dec_input = decoder_input[:,t] 
            else: 
                dec_input = output 
        
        return outputs 
    
encoder = Encoder(input_dim = X_train.shape[-1], hidden_dim = hidden_dim, n_layers = n_layers, dropout=dropout) 
attention = BahdanauAttention(dec_output_dim=hidden_dim, units=hidden_dim) 
decoder = Decoder(
    dec_feature_size = target_n, 
    encoder_hidden_dim=hidden_dim, 
    output_dim=target_n, 
    decoder_hidden_dim=hidden_dim, 
    n_layers=n_layers, 
    dropout=dropout, 
    attention=attention 
)

model = Seq2Seq(encoder, decoder, attention) 
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
criterion = nn.L1Loss() 

def train_step(batch_item, epoch, batch, training, teacher_forcing): 
    encoder_input = batch_item["encoder_input"].to(device) 
    decoder_input = batch_item["decoder_input"].to(device) 
    if training is True: 
        model.train() 
        optimizer.zero_grad() 
        output = model(encoder_input, decoder_input, teacher_forcing) 
        output = output.squeeze(dim=2) 
        loss = criterion(output, decoder_input[:, 1:]) 
        loss.backward() 
        optimizer.step() 
        return loss  
    else:
        model.eval() 
        with torch.no_grad(): 
            output = model(encoder_input, decoder_input, False) 
            output = output.squeeze(dim=2) 
            loss = criterion(output, decoder_input[:, 1:]) 
        return loss
    
best_val_loss = 999999999999999999999

for epoch in tqdm(range(EPOCHS), position=0, leave=True, desc="EPOCHS"): 
    print("epoch {}".format(epoch)) 
    total_loss, total_val_loss = 0, 0 
    total_score, total_val_score = 0, 0 
    tqdm_dataset = tqdm(enumerate(train_dataloader)) 
    training = True 
    for batch, batch_item in tqdm_dataset: 
        batch_loss = train_step(batch_item, epoch, batch, training, teacher_forcing) 
        total_loss += batch_loss.item() 
        tqdm_dataset.set_postfix({
            "Epoch": epoch + 1,
            "Loss": "{:06f}".format(batch_loss.item()), 
            "Total Loss": "{:06f}".format(total_loss / (batch+1))
        })
    
    tqdm_dataset = tqdm(enumerate(val_dataloader)) 
    training = False 
    for batch, batch_item in tqdm_dataset: 
        batch_loss = train_step(batch_item, epoch, batch, training, teacher_forcing) 
        total_val_loss += batch_loss.item()  
        tqdm_dataset.set_postfix({
            "Epoch": epoch + 1, 
            "Val Loss": "{:06f}".format(batch_loss.item()), 
            "Total Val Loss": "{:06f}".format(total_val_loss / (batch+1)) 
        }) 
    
    avg_val_loss = total_val_loss / (batch+1) 
    
    if avg_val_loss < best_val_loss: 
        best_val_loss = avg_val_loss 
        print(f"cur best val loss: {best_val_loss}") 
        print("saving best checkpoint") 
        torch.save(model, "seq2seq_best.pt") 
