import time
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import telegram
from ccxt.base.errors import ExchangeError
from ccxt.bybit import bybit as BybitExchange
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup, Comment
from dateutil import parser
from pybit import HTTP
import pprint
from transformers import *
import math
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmakr = True

seed_everything(7789)


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

class NeuralCLF(nn.Module):
        def __init__(self, chart_features, sequence_length, d_model, num_classes, n_heads, num_encoders):
                super(NeuralCLF, self).__init__()
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
                self.fc = nn.Linear(self.d_model, self.num_classes)
                self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc)
        def forward(self, x):
                x = self.chart_embedder(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = self.attentive_pooling(x)
                x = self.multi_dropout(x)
                return x

class BTC_Encoder:
    def __init__(self, symbol, bybit_credential, telegram_credential, high_chkpt, low_chkpt):
        self.exchange: BybitExchange = ccxt.bybit({
            'enableRateLimit': True,
            'apiKey': bybit_credential["api_key"],
            'secret': bybit_credential["api_secret"]
        })
        self.symbol = symbol
        self.exchange.load_markets()
        self.symbol_id = self.exchange.market(self.symbol)["id"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.high_encoder = NeuralCLF(chart_features=15, sequence_length=42, d_model=256, num_classes=1, n_heads=8, num_encoders=6)
        print(self.high_encoder.load_state_dict(torch.load(high_chkpt, map_location=self.device)))
        self.high_encoder.to(self.device)
        self.high_encoder.eval()

        self.low_encoder = NeuralCLF(chart_features=15, sequence_length=42, d_model=256, num_classes=1, n_heads=8, num_encoders=6)
        print(self.low_encoder.load_state_dict(torch.load(low_chkpt, map_location=self.device)))
        self.low_encoder.to(self.device)
        self.low_encoder.eval()

        self.telebot = telegram.Bot(token=telegram_credential["token"])
        self.telegram_chat_id = telegram_credential["chat_id"]


        self.session = HTTP(endpoint="https://api.bybit.com",
                            api_key=bybit_credential["api_key"],
                            api_secret =bybit_credential["api_secret"],
                            spot=False)
    def get_df(self):
        df = pd.DataFrame(self.exchange.fetch_ohlcv(self.symbol, timeframe="4h", limit=200))
        df = df.rename(columns={0:"timestamp",
                                1:"open",
                                2:"high",
                                3:"low",
                                4:"close",
                                5:"volume"})
        return df

    def create_timestamps(self, df):
        dates = df['timestamp'].values
        timestamp = []
        for i in range(len(dates)):
            date_string = self.exchange.iso8601(int(dates[i]))
            date_string = date_string[:10] + " " + date_string[11:-5]
            timestamp.append(date_string)
        df['datetime'] = timestamp
        df = df.drop(columns={'timestamp'})
        df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
        return df

    @staticmethod
    def preprocess_seq_data(chart_df):
        openv = chart_df["open"].values
        close = chart_df["close"].values
        high = chart_df["high"].values
        low = chart_df["low"].values
        volume = chart_df["volume"].values
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
        return chart_df[["bop", "ebsw", "cmf", "rsi/100", "r_open", "r_close", "r_high", "r_low", "r_volume", "high/low", "high/open", "low/open", "close/open", "high/close", "low/close"]]

    def send_message(self, text):
        try:
            self.telebot.sendMessage(chat_id=self.telegram_chat_id, text=text)
        except Exception as e:
            print(e)

    def my_floor(self, a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    def get_position_size(self):
        pos = self.session.my_position(symbol="BTCUSDT")
        long_size = pos['result'][0]['size']
        short_size = pos['result'][1]['size']
        return max(long_size, short_size)

    def get_best_bid_ask(self):
        orderbook = self.exchange.fetch_order_book(symbol=self.symbol)
        max_bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
        min_ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
        return max_bid, min_ask

    def place_best_buy_limit_order(self, qty, reduce_only, stop_loss, take_profit):
        max_bid, min_ask = self.get_best_bid_ask()
        resp = self.session.place_active_order(
            symbol=self.symbol_id,
            side="Buy",
            order_type="Limit",
            qty=qty,
            price=min_ask,
            time_in_force="GoodTillCancel",
            reduce_only=reduce_only,
            close_on_trigger=False,
            stop_loss = stop_loss,
            take_profit = take_profit)

    def place_best_sell_limit_order(self, qty, reduce_only, stop_loss, take_profit):
        max_bid, min_ask = self.get_best_bid_ask()
        resp = self.session.place_active_order(
            symbol=self.symbol_id,
            side="Sell",
            order_type="Limit",
            qty=qty,
            price=max_bid,
            time_in_force="GoodTillCancel",
            reduce_only=reduce_only,
            close_on_trigger=False,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def execute_trade(self):
        iterations = 0
        move = 0
        seq_len = 42
        while True:
            self.send_message(f"===== Trade Iteration {str(iterations)} =====")
            print(f"===== Trade Iteration {str(iterations)} =====")
            t0 = time.time()
            df = self.get_df()
            df = self.create_timestamps(df)
            df = self.preprocess_seq_data(df)
            X = df.iloc[df.shape[0]-seq_len-1:-1].values
            X = torch.tensor(X).float()
            X = torch.reshape(X, (-1, 42, 15)).to(self.device)
            with torch.no_grad():
                high_prediction = self.high_encoder(X)
                low_prediction = self.low_encoder(X)
                high_prediction = high_prediction.detach().cpu().numpy().flatten()[0]
                low_prediction = low_prediction.detach().cpu().numpy().flatten()[0]

            # 포지션 정리
            qty = self.get_position_size()
            if qty == 0:
                self.send_message("currently no positions opened, so we do not need to close any")
            else:
                if move == -1:
                    self.place_best_buy_limit_order(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)
                elif move == 1:
                    self.place_best_sell_limit_order(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)

            max_bid, min_ask = self.get_best_bid_ask()
            cur_price = (max_bid + min_ask) / 2.0
            balances = self.exchange.fetch_balance({"coin":"USDT"})["info"]
            usdt = balances["result"]["list"][0]["availableBalance"]
            self.send_message("current cash status = " + str(usdt))
            qty = float(usdt) / float(cur_price)
            qty = self.my_floor(qty, precision=5)

            print(high_prediction, low_prediction)

            if np.abs(high_prediction) >= np.abs(low_prediction):
                take_profit = float(cur_price) * (1+high_prediction)
                take_profit = round(take_profit)
                self.place_best_buy_limit_order(
                    qty = qty,
                    reduce_only = False,
                    stop_loss = None,
                    take_profit = take_profit)
                move = 1
            else:
                take_profit = float(cur_price) * (1+low_prediction)
                take_profit = round(take_profit)
                self.place_best_sell_limit_order(
                    qty = qty,
                    reduce_only=False,
                    stop_loss = None,
                    take_profit = take_profit)
                move = -1

            iterations += 1
            self.send_message("waiting for the next 20 seconds \(^.^)/")
            elapsed = time.time() - t0
            time.sleep(20  - elapsed)

bybit_cred = {
    "api_key": "JiIIWS3V0ZBiOJCaSJ",
    "api_secret": "kEBfigCJcgNpvnFxN0dx9rwJI8XjKjJoIKhS",
}
telegram_cred = {
    "token": "5322673870:AAHO3hju4JRjzltkG5ywAwhjaPS2_7HFP0g",
    "chat_id": 1720119057,
}
trader = BTC_Encoder(
    symbol="BTCUSDT",
    bybit_credential=bybit_cred,
    telegram_credential=telegram_cred,
    high_chkpt="HighEncoder_best_val_loss_0.00012690026269410737.pt",
    low_chkpt="LowEncoder_best_val_loss:0.000143110312637873"
)
trader.execute_trade()
