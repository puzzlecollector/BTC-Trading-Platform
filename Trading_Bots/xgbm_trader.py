import time 
import ccxt 
import numpy as np 
import pandas as pd 
import pandas_ta as ta 
import telegram
import time 
import random 
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ccxt.base.errors import ExchangeError 
from ccxt.bybit import bybit as BybitExchange 
from datetime import datetime, timedelta 
from transformers import XLMRobertaForSequenceClassification 
from tokenization_roberta_spm import FairSeqRobertaSentencePieceTokenizer
import requests 
from bs4 import BeautifulSoup, Comment 
from dateutil import parser 
from pybit import HTTP 
import pprint
import pickle 
import joblib 
from xgboost import XGBClassifier 


class CBITS_XGBM_Module:
    TAKE_PROFIT_PERCENT = 0.7 # fixed 
    STOP_LOSS_PERCENT = 3.0 # fixed - we are pretty much assuming that there is no stop loss by setting it to be a large value 
    leverage = 1 # fixed 
    def __init__(self, symbol, bybit_credential, telegram_credential, long_model_chkpt, short_model_chkpt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.exchange: BybitExchange = ccxt.bybit({
            "enableRateLimit": True, 
            "apiKey": bybit_credential["api_key"],
            "secret": bybit_credential["api_secret"]
        }) 
        self.exchange.load_markets() 
        self.symbol = symbol 
        self.symbol_id = self.exchange.market(self.symbol)["id"]
        self.xgbm_long = XGBClassifier()
        self.xgbm_long.load_model(long_model_chkpt) 
        self.xgbm_short = XGBClassifier()
        self.xgbm_short.load_model(short_model_chkpt) 
        self.telebot = telegram.Bot(token=telegram_credential["token"]) 
        self.telegram_chat_id = telegram_credential["chat_id"] 
        self.tokenizer = FairSeqRobertaSentencePieceTokenizer.from_pretrained("fairseq-roberta-all-model") 
        self.LM = XLMRobertaForSequenceClassification.from_pretrained("totoro4007/cryptoroberta-base-finetuned") 
        self.softmax = nn.Softmax(dim=1) 
        self.session = HTTP(endpoint="https://api.bybit.com",
                            api_key=bybit_credential["api_key"],
                            api_secret=bybit_credential["api_secret"],
                            spot=False)

    def get_df(self):
        df = pd.DataFrame(self.exchange.fetch_ohlcv(self.symbol, timeframe="4h", limit=200)) 
        df = df.rename(columns={0: "timestamp", 
                                1: "open", 
                                2: "high", 
                                3: "low",
                                4: "close", 
                                5: "volume"}) 
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
        return df 

    @staticmethod 
    def preprocess_data_for_CBITS(df): 
        hours, days, months, years = [], [], [], [] 
        for dt in tqdm(df["datetime"]): 
            hour = pd.to_datetime(dt).hour 
            day = pd.to_datetime(dt).day 
            month = pd.to_datetime(dt).month 
            year = pd.to_datetime(dt).year 
            hours.append(hour) 
            days.append(day) 
            months.append(month) 
            years.append(year) 
        df["hour"], df["day"], df["month"], df["year"] = hours, days, months, years 
        df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True) 
        df["bop"] = df.ta.bop(lookahead=False) 
        df["ebsw"] = df.ta.ebsw(lookahead=False) 
        df["cmf"] = df.ta.cmf(lookahead=False) 
        df["rsi"] = df.ta.rsi(lookahead=False) 
        df["vwap"] = df.ta.vwap(lookahead=False)
        df["vwap/open"] = df["vwap"] / df["open"] 
        df["high/low"] = df["high"] / df["low"] 
        df["close/open"] = df["close"] / df["open"] 
        df["high/open"] = df["high"] / df["open"] 
        df["low/open"] = df["low"] / df["open"] 
        for l in range(1,12): 
            for col in ["high", "low", "volume", "vwap"]:
                val = df[col].values 
                val_ret = [None for _ in range(l)] 
                for i in range(l, len(val)): 
                    if val[i-l] == 0: 
                        ret = 1 
                    else: 
                        ret = val[i] / val[i-l]
                    val_ret.append(ret) 
                df["{}_change_{}".format(col, l)] = val_ret 
        df = df.dropna() 
        df = df.drop(columns={"datetime", "open", "high", "low", "close", "volume", "vwap", "year"})
        return df


    def send_message(self, text): 
        try: 
            self.telebot.sendMessage(chat_id=self.telegram_chat_id, text=text) 
        except Exception as e: 
            print(e) 

    def get_position_size(self): 
        pos = self.session.my_position(symbol=self.symbol) 
        long_size = pos["result"][0]["size"] 
        short_size = pos["result"][1]["size"] 
        return max(long_size, short_size) 

    def get_best_bid_ask(self): 
        orderbook = self.exchange.fetch_order_book(symbol=self.symbol) 
        max_bid = orderbook["bids"][0][0] if len(orderbook["bids"]) > 0 else None 
        min_ask = orderbook["asks"][0][0] if len(orderbook["asks"]) > 0 else None 
        return max_bid, min_ask 

    def place_best_buy_limit_order(self, qty, reduce_only, stop_loss, take_profit): 
        max_bid, min_ask = self.get_best_bid_ask() 
        resp = self.session.place_active_order(
            symbol = self.symbol_id, 
            side = "Buy",
            order_type = "Limit", 
            qty = qty, 
            price = min_ask, 
            time_in_force = "GoodTillCancel", 
            reduce_only = reduce_only, 
            close_on_trigger = False, 
            stop_loss = stop_loss, 
            take_profit = take_profit
        )

    def place_best_sell_limit_order(self, qty, reduce_only, stop_loss, take_profit): 
        max_bid, min_ask = self.get_best_bid_ask() 
        resp = self.session.place_active_order(
            symbol = self.symbol_id, 
            side = "Sell", 
            order_type = "Limit", 
            qty = qty, 
            price = max_bid, 
            time_in_force = "GoodTillCancel", 
            reduce_only = reduce_only, 
            close_on_trigger = False, 
            stop_loss = stop_loss, 
            take_profit = take_profit 
        ) 

    def get_articles(self, headers, url): 
        news_req = requests.get(url, headers=headers)
        soup = BeautifulSoup(news_req.content, "lxml") 
        title = soup.find("p", {"class":"ArticleBigTitle"}).text.strip() 
        content = soup.find("div", {"class": "viewArticle"}).text.strip() 
        return title, content 

    def time_in_range(self, start, end, x):  
        if start <= end: 
            return start <= x <= end 
        else: 
            return start <= x or x <= end 

    def my_floor(self, a, precision=0): 
        return np.true_divide(np.floor(a* 10**precision), 10**precision) 

    def execute_trade(self): 
        # run trade (8 hours cycle: 1 -> 9 -> 5 -> 1 -> ...)
        t0 = time.time() 
        iteration = 0 
        move = 0 # -1: short, 1: long, 0: hold (if hold option exists) 
        self.LM.cuda() # test running on V100-32GB  
        self.LM.eval() 
        MAX_LOOKBACK = 1000 

        while True: 
            self.send_message(f"==== Trade Iteration {str(iteration)} ====") 
            t0 = time.time() 
            df = self.get_df() 
            df = self.create_timestamps(df) 
            df = self.preprocess_data_for_CBITS(df) 
            # collect news information 
            titles, contents = [], [] 
            stop = False 
            # get time information 
            cur_time = datetime.utcnow() 
            cur_time = str(cur_time) 
            dt_tm_utc = datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S.%f") 
            tm_kst = dt_tm_utc + timedelta(hours=9) 
            start = tm_kst - timedelta(hours=4) 
            end = tm_kst 
            for i in range(1, MAX_LOOKBACK): 
                if stop == True: 
                    break 
                try: 
                    links, times = [], [] 
                    # start web scraping 
                    url = "https://www.tokenpost.kr/coinness?page=" + str(i) 
                    headers = requests.utils.default_headers() 
                    headers.update({
                        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0",
                    })
                    news_req = requests.get(url, headers=headers) 
                    soup = BeautifulSoup(news_req.content, "lxml") 
                    elems = soup.find_all("div", {"class":"listWrap"})
                    for e in tqdm(elems, position=0, leave=True): 
                        # collect articles on the current page 
                        for child_div in e.find_all("div", {"class":"articleListWrap paddingT15 paddingB15"}):
                            for a in child_div.find_all("a"):
                                l = a["href"]  
                                if "/article-" in l: 
                                    links.append("https://www.tokenpost.kr" + str(l))
                        # collect date time info
                        for child_div in e.find_all("span", {"class":"articleListDate marginB8"}):
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
                                print("Error occurred when getting article content!") 
                                print(e) 
                        elif date < start: 
                            stop = True 
                            break 
                except Exception as e: 
                    print("Error occurred when scraping!") 
                    print(e) 
                time.sleep(1.0) 
                
            
            self.send_message("{} coinness news articles collected in the designated timeframe".format(len(titles))) 

            sentiment_scores = torch.tensor([0,0,0]) 
            if len(titles) > 0: 
                for i in range(len(titles)): 
                    encoded_inputs = self.tokenizer(titles[i], contents[i], return_tensors="pt", max_length=512, truncation=True).to(self.device)
                    with torch.no_grad():
                        output = self.LM(**encoded_inputs) 
                        logits = output["logits"] 
                        probs = self.softmax(logits) 
                        probs = probs.detach().cpu().numpy().flatten()
                        sentiment_scores += probs 
            sentiment_probs = nn.Softmax()(torch.tensor([sentiment_scores[0], sentiment_scores[1]]))
            sentiment_probs = np.array(sentiment_probs).reshape((1,2))
            x = df.values[-2].reshape((-1, df.shape[1]))
            x = np.concatenate([x, sentiment_probs], axis=1) 

            take_long = self.xgbm_long.predict(x)[0]
            take_long_confidence = self.xgbm_long.predict_proba(x)[0][1]  
            take_short = self.xgbm_short.predict(x)[0]  
            take_short_confidence = self.xgbm_short.predict_proba(x)[0][1] 

            if take_long == 1 and take_short == 0: 
                pred_class = 0 
            elif take_long == 0 and take_short == 1: 
                pred_class = 1 
            else: 
                if take_long_confidence >= take_short_confidence: 
                    pred_class = 0 
                else:
                    pred_class = 1 

            
            pos_dict = {0:"Long", 1:"Short", 2:"Hold"} 
            if pred_class == 0: 
                self.send_message(f"Directional prediction {pos_dict[pred_class]}, probability {take_long_confidence}") 
            elif pred_class == 1: 
                self.send_message(f"Directional prediction {pos_dict[pred_class]}, probability {take_short_confidence}")  
            self.send_message(f"Current News Sentiment Score Towards BTC (Positive, Negative): {sentiment_probs[0]}") 
            
            if pred_class == 0.0: 
                qty = self.get_position_size() 
                if qty == 0: 
                    self.send_message("currently no positions opened, so we do not need to close any") 
                else: 
                    if move == -1: 
                        self.send_message("closing previous short position and opening long position") 
                        self.place_best_buy_limit_order(qty=qty, reduce_only=True, stop_loss=None, take_profit=None) 
                    elif move == 1: 
                        self.send_message("closing previous long position and opening long position") 
                        self.place_best_sell_limit_order(qty=qty, reduce_only=True, stop_loss=None, take_profit=None) 
                max_bid, min_ask = self.get_best_bid_ask() 
                cur_price = (max_bid + min_ask) / 2.0 
                balances = self.exchange.fetch_balance({"coin":"USDT"})["info"] 
                usdt = balances["result"]["USDT"]["available_balance"]
                self.send_message("current cash status = " + str(usdt)) 
                qty = float(usdt) / float(cur_price) * self.leverage 
                qty = self.my_floor(qty, precision=5) 
                stop_loss = cur_price * (1-self.STOP_LOSS_PERCENT/100) 
                stop_loss = round(stop_loss) 
                take_profit = cur_price * (1+self.TAKE_PROFIT_PERCENT/100) 
                take_profit = round(take_profit) 
                try: 
                    self.place_best_buy_limit_order(qty=qty, reduce_only=False, stop_loss=stop_loss, take_profit=take_profit) 
                except Exception as e: 
                    self.send_message(e) 
                    self.send_message("error occurred while trying to open long position") 
                move = 1 
            elif pred_class == 1.0: 
                qty = self.get_position_size() 
                if qty == 0: 
                    self.send_message("currently no positions opened, so we do not need to close any.") 
                else: 
                    if move == -1: 
                        self.send_message("closing previous short position and opening short position") 
                        self.place_best_buy_limit_order(qty=qty, reduce_only=True, stop_loss=None, take_profit=None) 
                    elif move == 1: 
                        self.send_message("closing previous long position and opening short position")
                        self.place_best_sell_limit_order(qty=qty, reduce_only=True, stop_loss=None, take_profit=None) 
                max_bid, min_ask = self.get_best_bid_ask() 
                cur_price = (max_bid + min_ask) / 2.0 
                balances = self.exchange.fetch_balance({"coin":"USDT"})["info"] 
                usdt = balances["result"]["USDT"]["available_balance"] 
                self.send_message("current cash status = " + str(usdt)) 
                qty = float(usdt) / float(cur_price) * self.leverage 
                qty = self.my_floor(qty, precision=5) 
                take_profit = float(cur_price) * (1-self.TAKE_PROFIT_PERCENT/100) 
                take_profit = round(take_profit) 
                stop_loss = float(cur_price) * (1+self.STOP_LOSS_PERCENT/100) 
                stop_loss = round(stop_loss) 
                try: 
                    self.place_best_sell_limit_order(qty=qty, reduce_only=False, stop_loss=stop_loss, take_profit=take_profit) 
                except Exception as e: 
                    self.send_message(e) 
                    self.send_message("error occurred while trying to open short position") 
                move = -1 
            iteration += 1 
            self.send_message("waiting for the next 4 hours \(*.*)/") 
            elapsed = time.time() - t0 
            time.sleep(60*60*4 - elapsed) 


if __name__=="__main__":
    bybit_cred = {
        "api_key": "Ef6vTSJM17IRYMRRCj",
        "api_secret": "Z2m4YyMorq4rg7GbxLcU4lJkA38GTwuPvlj1",
    }
    telegram_cred = {
        "token":"5322673870:AAHO3hju4JRjzltkG5ywAwhjaPS2_7HFP0g",
        "chat_id":1720119057,
    }
    trader = CBITS_XGBM_Module(
        symbol = "BTCUSDT",
        bybit_credential = bybit_cred, 
        telegram_credential = telegram_cred, 
        long_model_chkpt = "XGBoost_4hours_long",
        short_model_chkpt = "XGBoost_4hours_short"
    )
    time.sleep(60*91)
    trader.execute_trade()
