import time
from datetime import datetime, timedelta
import ccxt # pip install ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta # pip install pandas-ta
import telegram # pip install python-telegram-bot
from ccxt.base.errors import ExchangeError
from ccxt.bybit import bybit as BybitExchange
from binance.client import Client # pip install python-binance
from binance.helpers import round_step_size
import matplotlib.pyplot as plt
import lightgbm as lgbm # pip install lightgbm
from xgboost import XGBClassifier # pip install xgboost
from pytorch_tabnet.tab_model import TabNetClassifier # pip install pytorch-tabnet
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AlbertTokenizer # pip install transformers
import torch # pip install torch
import torch.nn as nn
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup, Comment # pip install beautifulsoup4
from dateutil import parser

class Ensemble_Trader:
    stop_percent = 2.0
    take_percent = 1.0
    def __init__(self, binance_cred, telegram_cred, xgboost_ckpt, lgbm_ckpt, tabnet_ckpt):
        # ccxt binance object
        self.exchange: BinanceExchange = ccxt.binance({
            "enableRateLimit": True,
            "apiKey": binance_cred["api_key"],
            "secret": binance_cred["api_secret"]
        })
        # python-binance object
        self.client = Client(api_key=binance_cred["api_key"],
                             api_secret=binance_cred["api_secret"])
        # load model checkpoints
        self.xgboost = XGBClassifier()
        self.xgboost.load_model(xgboost_ckpt)
        self.lgbm = joblib.load(lgbm_ckpt)
        self.tabnet = TabNetClassifier()
        self.tabnet.load_model(tabnet_ckpt)
        # define telegram object
        self.telebot = telegram.Bot(token=telegram_cred["token"])
        self.telegram_chat_id = telegram_cred["chat_id"]
        self.symbol = "BTCUSDT"
        # load DeBERTa model for sentiment score calculation: model is run on CPU as we are assuming gpu is unavailable
        self.tokenizer = AlbertTokenizer.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned")
        self.LM = AutoModelForSequenceClassification.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned")
        self.softmax = nn.Softmax(dim=1)

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
        df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
        return df

    def my_floor(self, a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    def get_tick_size(self, symbol: str) -> float:
        info = self.client.futures_exchange_info()
        for symbol_info in info['symbols']:
            if symbol_info['symbol'] == symbol:
                for symbol_filter in symbol_info['filters']:
                    if symbol_filter['filterType'] == 'PRICE_FILTER':
                        return float(symbol_filter['tickSize'])

    def get_rounded_price(self, symbol: str, price: float) -> float:
        return round_step_size(price, self.get_tick_size(symbol))

    @staticmethod
    def preprocess_data(chart_df):
        close_prices = chart_df["close"].values
        hours, days, months, years = [],[],[],[]
        for dt in tqdm(chart_df['datetime']):
            hour = pd.to_datetime(dt).hour
            day = pd.to_datetime(dt).day
            month = pd.to_datetime(dt).month
            year = pd.to_datetime(dt).year
            hours.append(hour)
            days.append(day)
            months.append(month)
            years.append(year)
        chart_df['hour'] = hours
        chart_df['day'] = days
        chart_df['month'] = months
        chart_df['year'] = years
        chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)
        ### addition of chart features ###
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
        chart_df["sma"] = chart_df.ta.sma(lookahead=False)
        chart_df["sma/close"] = chart_df["sma"] / chart_df["close"]
        ### addition of recent differenced features ###
        for l in tqdm(range(1, 12), position=0, leave=True):
            for col in ["high", "low", "volume", "vwap"]:
                val = chart_df[col].values
                val_ret = [None for _ in range(l)]
                for i in range(l, len(val)):
                    if val[i-l] == 0:
                        ret = 1
                    else:
                        ret = val[i] / val[i-l]
                    val_ret.append(ret)
                chart_df["{}_change_{}".format(col, l)] = val_ret

        ### drop unnecessary columns ###
        chart_df.drop(columns={"datetime","year","open","high","low","close","volume","vwap","hwma","linreg", "sma"}, inplace=True)
        chart_df.dropna(inplace=True)
        return chart_df, close_prices

    def send_message(self, text):
        try:
            self.telebot.sendMessage(chat_id=self.telegram_chat_id, text=text)
        except Exception as e:
            print(e)

    def get_position_size(self):
        positions = self.client.futures_account()["positions"]
        for p in positions:
            if p["symbol"] == self.symbol:
                amt = p["positionAmt"]
                return np.abs(float(amt))

    def get_best_bid_ask(self):
        orderbook = self.client.get_order_book(symbol=self.symbol)
        max_bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
        min_ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
        return max_bid, min_ask

    def place_best_buy_limit_order(self, reduce_only, qty, stopPrice, targetPrice):
        futures_order = self.client.futures_create_order(
            symbol=self.symbol,
            side="BUY",
            quantity=qty,
            reduceOnly=reduce_only,
            type="MARKET")
        if reduce_only == False: # send in stop loss and take profit
            futures_stop_loss = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="SELL",
                type="STOP_MARKET",
                stopPrice=stopPrice,
                closePosition=True)
            futures_take_profit = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="SELL",
                type="TAKE_PROFIT_MARKET",
                stopPrice=targetPrice,
                closePosition=True)
            stoploss_id = futures_stop_loss['orderId']
            takeprofit_id = futures_take_profit['orderId']
            return stoploss_id, takeprofit_id
        else:
            return None, None

    def place_best_sell_limit_order(self, reduce_only, qty, stopPrice, targetPrice):
        futures_order = self.client.futures_create_order(
            symbol=self.symbol,
            side="SELL",
            quantity=qty,
            reduceOnly=reduce_only,
            type="MARKET")
        if reduce_only == False: # send in take profit and stop loss
            futures_stop_loss = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="BUY",
                type="STOP_MARKET",
                stopPrice=str(stopPrice),
                closePosition=True)
            futures_take_profit = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="BUY",
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(targetPrice),
                closePosition=True)
            stoploss_id = futures_stop_loss['orderId']
            takeprofit_id = futures_take_profit['orderId']
            return stoploss_id, takeprofit_id
        else:
            return None, None

    # utilities for scraping coinness news article real time from the tokenpost website
    def get_articles(self, headers, url):
        news_req = requests.get(url, headers=headers)
        soup = BeautifulSoup(news_req.content, "lxml")
        title = soup.find("p",{"class":"ArticleBigTitle"}).text.strip()
        content = soup.find("div", {"class":"viewArticle"}).text.strip()
        return title, content

    def time_in_range(self, start, end, x):
        if start <= end:
            return start <= x <= end
        else:
            return start <= x or x <= end

    # trading logic
    def execute_trade(self):
        # run trade (4 hour cycle: 1 -> 5 -> 9 -> 1 ...)
        iteration = 0
        move = 0 # -1: short, 1: long
        stoploss_id, takeprofit_id = -1, -1 # stop loss and take profit id, close them after a single iteration just in case they are not filled
        leverage = 1
        self.LM.eval()
        MAX_LOOKBACK = 1000 # number of pages to explore when scraping news
        while True:
            self.send_message(f"==== Trade Iteration {str(iteration)} ====")
            t0 = time.time()
            df = self.get_df()
            df = self.create_timestamps(df)
            df, close = self.preprocess_data(df)
            prev_close = close[-2]
            # collect news information
            titles, contents = [], []
            stop = False
            # get time information
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
                    # start web scraping
                    url = "https://www.tokenpost.kr/coinness?page=" + str(i)
                    headers = requests.utils.default_headers()
                    headers.update({
                        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
                    })
                    news_req = requests.get(url, headers=headers)
                    soup = BeautifulSoup(news_req.content,"lxml")
                    elems = soup.find_all("div", {"class":"listWrap"})
                    for e in tqdm(elems, position=0, leave=True):
                        # collect articles on the current page
                        for child_div in e.find_all("div", {"class":"articleListWrap paddingT15 paddingB15"}):
                            for a in child_div.find_all("a"):
                                l = a['href']
                                if '/article-' in l:
                                    links.append('https://www.tokenpost.kr' + str(l))
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
                                print("Error occured with getting article content!")
                                print(e)
                        elif date < start:
                            stop = True
                            break
                        time.sleep(1.0)
                except Exception as e:
                    print("Error occured with scraping!")
                    print(e)
                time.sleep(1.0)
            self.send_message("{} coinness news articles collected in the designated timeframe".format(len(titles)))
            sentiment_scores = torch.tensor([0, 0, 0])
            if len(titles) > 0:
                for i in range(len(titles)):
                    encoded_inputs = self.tokenizer(titles[i],contents[i],return_tensors="pt",max_length=512,truncation=True)
                    with torch.no_grad():
                        output = self.LM(**encoded_inputs)
                        logits = output['logits']
                        probs = self.softmax(logits)
                        probs = probs.detach().cpu().numpy().flatten()
                        sentiment_scores += probs
            x = df.values[-2].reshape((-1, df.shape[1]))
            sentiment_scores = sentiment_scores[:2]
            sentiment_scores = nn.Softmax()(sentiment_scores)
            sentiment_scores = sentiment_scores.numpy().reshape((1,2))
            x = np.concatenate([x, sentiment_scores], axis=1)
            xgboost_probs = self.xgboost.predict_proba(x)
            lgbm_probs = self.lgbm.predict_proba(x)
            tabnet_probs = self.tabnet.predict_proba(x)
            avg_probs = (xgboost_probs + lgbm_probs + tabnet_probs) / 3.0
            pred = np.argmax(avg_probs, axis=1)[0]
            max_prob = np.max(avg_probs)
            pos_dict = {0:'Long', 1:'Short', 2:'Hold'}
            self.send_message("Bot's directional prediction {}, model confidence {:.3f}".format(pos_dict[pred], max_prob))
            self.send_message("current BTC news sentiment score (positive, negative): {}".format(sentiment_scores))

            if iteration > 0:
                # get rid of unclosed take profit and stop loss orders
                try:
                    self.client.futures_cancel_order(symbol=self.symbol, orderId=stoploss_id)
                except Exception as e:
                    print(e)
                try:
                    self.client.futures_cancel_order(symbol=self.symbol, orderId=takeprofit_id)
                except Exception as e:
                    print(e)
                # close if there are any positions open from the previous iteration
                qty = self.get_position_size()
                print("quantity = {}".format(qty))
                if qty == 0:
                    self.send_message("no positions open... stop loss or take profit was probably triggered.")
                else:
                    if move == -1:
                        self.send_message("Closing previous short position...")
                        self.place_best_buy_limit_order(reduce_only=True, qty=qty, stopPrice=None, targetPrice=None)
                    elif move == 1:
                        self.send_message("Closing previous long position...")
                        self.place_best_sell_limit_order(reduce_only=True, qty=qty, stopPrice=None, targetPrice=None)

            if pred == 0:
                btc_usdt = self.client.get_symbol_ticker(symbol=self.symbol)
                btc_usdt = float(btc_usdt['price'])
                stopPrice = prev_close * (1 - self.stop_percent / 100)
                targetPrice = prev_close * (1 + self.take_percent / 100)
                stopPrice = self.get_rounded_price(self.symbol, stopPrice)
                targetPrice = self.get_rounded_price(self.symbol, targetPrice)
                usdt = self.client.futures_account_balance()[6]["balance"] # get usdt balance
                self.send_message("current cash status = {}".format(usdt))
                qty = float(usdt) / float(btc_usdt)
                qty = self.my_floor(qty, precision=3)
                stoploss_id, takeprofit_id = self.place_best_buy_limit_order(reduce_only=False,
                                                                             qty=qty,
                                                                             stopPrice=stopPrice,
                                                                             targetPrice=targetPrice)
                move = 1
            elif pred == 1:
                btc_usdt = self.client.get_symbol_ticker(symbol=self.symbol)
                btc_usdt = float(btc_usdt['price'])
                stopPrice = prev_close * (1 + self.stop_percent / 100)
                targetPrice = prev_close * (1 - self.take_percent / 100)
                stopPrice = self.get_rounded_price(self.symbol, stopPrice)
                targetPrice = self.get_rounded_price(self.symbol, targetPrice)
                usdt = self.client.futures_account_balance()[6]["balance"] # get usdt balance
                self.send_message("current cash status = {}".format(usdt))
                qty = float(usdt) / float(btc_usdt)
                qty = self.my_floor(qty, precision=3)
                stoploss_id, takeprofit_id = self.place_best_sell_limit_order(reduce_only=False,
                                                                              qty=qty,
                                                                              stopPrice=stopPrice,
                                                                              targetPrice=targetPrice)
                move = -1

            iteration += 1
            self.send_message("waiting for the next 4 hours...")
            elapsed = time.time() - t0
            time.sleep(60*60*4 - elapsed)

### run trade ###
binance_cred = {
    "api_key": "86EkTZQqPf0zkC992tQ6PltR7ujXirZvEaxmXvIGUyvsenGWYfHcwbGKlJgjIDgR",
    "api_secret": "TvCuMxUYH3vuAmwLrLq5kSoryCXL47jsC1VLc8trpXjJwiyHqUsmcTWvGW6AbDaB"
}
telegram_cred = {
    "token": "5322673870:AAHO3hju4JRjzltkG5ywAwhjaPS2_7HFP0g",
    "chat_id":1720119057
}
trader = Ensemble_Trader(
    binance_cred = binance_cred,
    telegram_cred = telegram_cred,
    xgboost_ckpt="xgboost_btc_3",
    lgbm_ckpt="lgbm_btc.pkl",
    tabnet_ckpt="tabnet_btc.zip"
)

# 한국 시간으로 1시, 5시, 9시, 13시 ,17시, 21시 중 하나에 시작해야함
trader.execute_trade()
