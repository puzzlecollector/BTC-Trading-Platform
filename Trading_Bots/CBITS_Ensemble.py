import time 
import ccxt 
import numpy as np 
import pandas as pd
import pandas_ta as ta 
import telegram 
from ccxt.base.errors import ExchangError 
from ccxt.bybit import bybit as BybitExchange 
from tqdm import tqdm 
import torch 
import torch.nn as nn 
from datetime import datetime, timedelta 
from transformers import XLMRobertForSequenceClassification 
from tokenization_roberta_spm import FairSeqRobertaSentencePieceTokenizer
import requests 
from bs4 import BeautifulSoup, Comment 
from dateutil import parser 
from pybit import HTTP 
import pprint 
from lxml import etree 

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

class CBITS_Ensemble:
    STOP_LOSS_PERCENT = 2.0 
    def __init__(self, symbol, bybit_credential, telegram_credential, dnn_chkpt, xgb_chkpt): 
        self.exchange: BybitExchange = ccxt.bybit({
            'enableRateLimit': True,
            'apiKey': bybit_credential["api_key"],
            'secret': bybit_credential["api_secret"]
        })
        self.exchange.load_markets() 
        self.symbol = symbol 
        self.symbol_id = self.exchange.market(self.symbol)["id"] 
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.xgb = XGBClassifier() 
        self.xgb.load_model(xgb_chkpt) 
        
        self.dnn = DNN(num_classes=3, num_features=81) 
        self.chkpt = torch.load(dnn_chkpt) 
        self.dnn.load_state_dict(self.chkpt) 
        self.dnn.to(self.device) 
        self.dnn.eval() 
        
        self.telebot = telegram.Bot(token=telegram_credential["token"]) 
        self.telegram_chat_id = telegram_credential["chat_id"] 
        
        self.tokenizer = FairSeqRobertaSentencePieceTokenizer.from_pretrained("fairseq-roberta-all-model")
        self.roberta = XLMRobertaForSequenceClassification.from_pretrained("axiomlabs/KR-cryptoroberta-base")
        self.softmax = nn.Softmax(dim=1) 
        
        self.session = HTTP(endpoint = "https://api.bybit.com",
                            api_key = bybit_credential["api_key"],
                            api_secret = bybit_credential["api_secret"], 
                            spot = False) 
    
    def get_df(self): 
        df = pd.DataFrame(self.exchange.fetch_ohlcv(self.symbol, timeframe='4h', limit=200))
        df = df.rename(columns={0:"timestamp", 
                                1:"open",
                                2:"high", 
                                3:"low", 
                                4:"close",
                                5:"volume"}) 
        return df 

    def create_timestamps(self, df): 
        dates = df["timestamp"].values 
        timestamp = [] 
        for i in range(len(dates)): 
            date_string = self.exchange.iso8601(int(dates[i]))
            date_string = date_string[:10] + " " + date_string[11:-5] 
            timestamp.append(date_string)
        df["datetime"] = timestamp 
        df = df.drop(columns={"timestamp"}) 
        df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
        return df 
    
    @staticmethod 
    def preprocess_data_for_CBITS(chart_df): 
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
    
    def send_message(self, text): 
        try: 
            self.telebot.sendMessage(chat_id=self.telegram_chat_id, text=text) 
        except Exception as e: 
            print(e)
    
    def get_position_size(self): 
        pos = self.session.my_position(symbol="BTCUSDT") 
        long_size = pos["result"][0]["size"] 
        short_size = pos["result"][1]["size"] 
        return max(long_size, short_size) 
    
    def get_best_bid_ask(self): 
        orderbook = self.exchange.fetch_order_book(symbol=self.symbol)
        max_bid = orderbook["bids"][0][0] if len(orderbook["bids"]) > 0 else None
        min_ask = orderbook["asks"][0][0] if len(orderbook['asks']) > 0 else None
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
            take_profit = take_profit
        )
    
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
    
    def get_articles(self, headers, url): 
        news_req = request.get(url, headers=headers) 
        soup = BeautifulSoup(news_req.content, "lxml")
        title = soup.find("span",{"class":"view_top_title noselect"}).text.strip()
        dom = etree.HTML(str(soup))
        content = dom.xpath('//*[@id="articleContentArea"]/div[4]/div[1]/p/text()')[0]
        return title, content 

    def time_in_range(self, start, end, x): 
        if start <= end: 
            return start <= x <= end 
        else: 
            return start <= x or x <= end
    
    def my_floor(self, a, precision=0): 
        return np.true_divide(np.floor(a * 10**precision), 10**precision)
    
    def execute_trade(self): 
        iteration = 0 
        move = 0 # -1: short, 0: hold, 1: long 
        leverage = 1 # fixed 
        self.roberta.eval() 
        MAX_LOOKBACK = 300 
        while True: 
            self.send_message(f"==== Trade Iteration {str(iteration)} ====")
            t0 = time.time() 
            df = self.get_df() 
            df = self.create_timestamps(df) 
            df = self.preprocess_data_for_CBITS(df) 
            
            # collect news information
            titles, contents = [], [] 
            stop = False 
            cur_time = datetime.utcnow() 
            cur_time = str(cur_time) 
            dt_tm_utc = datetime.strptime(cur_time, '%Y-%m-%d %H:%M:%S.%f')
            tm_kst = dt_tm_utc + timedelta(hours=9)
            start = tm_kst - timedelta(hours=4) 
            end = tm_kst 
            for i in range(1, MAX_LOOKBACK): 
                if stop == True:
                    break 
                try:
                    links, times = [], [] 
                    url = "https://www.tokenpost.kr/coinness?page=" + str(i)
                    headers = requests.utils.default_headers()
                    headers.update({
                        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
                    })
                    news_req = requests.get(url, headers=headers)
                    soup = BeautifulSoup(news_req.content,"lxml")
                    elems = soup.find_all("div", {"class":"list_left_item"})
                    for e in tqdm(elems, position=0, leave=True, desc="Getting News Links"):
                        for child_div in e.find_all("div", {"class":"list_left_item_article"}):
                            for a in child_div.find_all("a"):
                                l = a['href']
                                if '/article-' in l:
                                    links.append('https://www.tokenpost.kr' + str(l))
                        # collect date time information 
                        for child_div in e.find_all("span", {"class":"day"}):
                            news_dt = parser.parse(child_div.text)
                            times.append(news_dt)
                    # only keep news updated within the time window 
                    for idx, date in enumerate(times): 
                        if self.time_in_range(start, end, date): 
                            try: 
                                title, content = self.get_articles(headers, links[idx])
                                titles.append(title) 
                                contents.append(content) 
                            except Exception as e: 
                                print("Error occurred with getting article content!") 
                                print(e) 
                        elif date < start:
                            stop = True 
                            break 
                        time.sleep(1.0) 
                except Exception as e:
                    print("Error occurred with scraping news links!") 
                    print(e) 
                time.sleep(1.0) 
            self.send_message("{} coinness news articles collected in the designated timeframe".format(len(titles)))
            
            # calculate sentiment scores 
            sentiment_scores = torch.tensor([0, 0, 0]) 
            if len(titles) > 0: 
                for i in range(len(titles)): 
                    encoded_inputs = self.tokenizer(str(titles[i]),
                                                    str(contents[i]),
                                                    return_tensors="pt",
                                                    max_length=512,
                                                    truncation=True)
                    with torch.no_grad(): 
                        output = self.roberta(**encoded_inputs)
                        logits = output['logits']
                        probs = self.softmax(logits)
                        probs = probs.detach().cpu().numpy().flatten()
                        sentiment_scores += probs
            x = df.values[-2].reshape((-1, df.shape[1])) 
            sentiment_scores = sentiment_scores[:2]
            sentiment_scores = nn.Softmax(dim=0)(sentiment_scores) 
            sentiment_scores = sentiment_scores.numpy().reshape((1, 2))
            x = np.concatenate([sentiment_scores, x], axis=1) 
            prob_xgb = self.xgb.predict_proba(x) 
            with torch.no_grad(): 
                x = torch.tensor(x).to(self.device) 
                prob_dnn = self.dnn(x) 
                prob_dnn = nn.Softmax(dim=1)(prob_dnn) 
                prob_dnn = prob_dnn.detach().cpu().numpy() 
            prob_avg = (prob_xgb + prob_dnn) / 2.0 
            action = np.argmax(prob_avg) 
            
            
            
            
            
                    

        
        








    
