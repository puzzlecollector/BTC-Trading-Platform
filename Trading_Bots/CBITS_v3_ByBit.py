import time
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import telegram
from ccxt.base.errors import ExchangeError
from ccxt.bybit import bybit as BybitExchange
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from transformers import XLMRobertaForSequenceClassification
from tokenization_roberta_spm import FairSeqRobertaSentencePieceTokenizer
import requests
from bs4 import BeautifulSoup, Comment
from dateutil import parser
from pybit import HTTP
import pprint

class CBITS_v3:
    STOP_LOSS_PERCENT = 7.5 # some large value - we do not need stop loss at 1x leverage

    def __init__(self, symbol, bybit_credential, telegram_credential, cbits_zip_dir, tabnet_high_dir, tabnet_low_dir):
        self.exchange: BybitExchange = ccxt.bybit({
            'enableRateLimit': True,
            'apiKey': bybit_credential["api_key"],
            'secret': bybit_credential["api_secret"]
        })
        self.exchange.load_markets()
        self.symbol = symbol
        self.symbol_id = self.exchange.market(self.symbol)["id"]
        self.set_leverage(buy_leverage=1, sell_leverage=1)

        self.tn = TabNetClassifier()
        self.tn.load_model(cbits_zip_dir)

        self.reg_high = TabNetRegressor()
        self.reg_high.load_model(tabnet_high_dir)

        self.reg_low = TabNetRegressor()
        self.reg_low.load_model(tabnet_low_dir)

        self.telebot = telegram.Bot(token=telegram_credential["token"])
        self.telegram_chat_id = telegram_credential["chat_id"]

        self.tokenizer = FairSeqRobertaSentencePieceTokenizer.from_pretrained("fairseq-roberta-all-model")
        self.LM = XLMRobertaForSequenceClassification.from_pretrained("totoro4007/cryptoroberta-base-finetuned")
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device('cuda')

        self.session = HTTP(endpoint="https://api.bybit.com",
                            api_key=bybit_credential["api_key"],
                            api_secret =bybit_credential["api_secret"],
                            spot=False)
    def get_df(self):
        df = pd.DataFrame(self.exchange.fetch_ohlcv(self.symbol, timeframe='4h', limit=200))
        df = df.rename(columns={0: 'timestamp',
                                1: 'open',
                                2: 'high',
                                3: 'low',
                                4: 'close',
                                5: 'volume'})
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
    def preprocess_data_for_CBITS(df):
        hours, days, months, years = [], [], [], []
        for dt in tqdm(df['datetime']):
            hour = pd.to_datetime(dt).hour
            day = pd.to_datetime(dt).day
            month = pd.to_datetime(dt).month
            year = pd.to_datetime(dt).year
            hours.append(hour)
            days.append(day)
            months.append(month)
            years.append(year)
        df['hour'], df['day'], df['month'], df['year'] = hours, days, months, years
        df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
        # add chart feature engineered data
        df['ebsw'] = df.ta.ebsw(lookahead=False)
        df['cmf'] = df.ta.cmf(lookahead=False)
        df['vwap'] = df.ta.vwap(lookahead=False)
        df['vwap/open'] = df['vwap'] / df['open']
        df['high/low'] = df['high'] / df['low']
        df['close/open'] = df['close'] / df['open']
        df['high/open'] = df['high'] / df['open']
        df['low/open'] = df['low'] / df['open']
        for l in range(1, 6):
            for col in ['open', 'high', 'low', 'close', 'volume']:
                val = df[col].values
                val_ret = [None for _ in range(l)]
                for i in range(l, len(val)):
                    if val[i-l] == 0:
                        ret = 1
                    else:
                        ret = val[i] / val[i-l]
                    val_ret.append(ret)
                df['{}_change_{}'.format(col, l)] = val_ret
        df = df.drop(columns={"datetime",
                              "open",
                              "high",
                              "low",
                              "close",
                              "volume",
                              "vwap",
                              "year"})
        return df

    @staticmethod
    def preprocess_data_for_HL(df):
        df.set_index(pd.DatetimeIndex(df['datetime']), inplace=True)
        df['bop'] = df.ta.bop(lookahead=False)
        df['ebsw'] = df.ta.ebsw(lookahead=False)
        df['cmf'] = df.ta.cmf(lookahead=False)
        df['rsi/100'] = df.ta.rsi(lookahead=False) / 100
        df['vwap'] = df.ta.vwap(lookahead=False)
        df['high/low'] = df['high'] / df['low']
        df['close/open'] = df['close'] / df['open']
        df['high/open'] = df['high'] / df['open']
        df['low/open'] = df['low'] / df['open']

        df['hwma'] = df.ta.hwma(lookahead=False)
        df['linreg'] = df.ta.linreg(lookahead=False)
        df['hwma/close'] = df['hwma'] / df['close']
        df['linreg/close'] = df['linreg'] / df['close']

        for i in range(1, 12):
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                val = df[col].values
                val_ret = [None for _ in range(i)]
                for j in range(i, len(val)):
                    if val[j-i] == 0:
                        ret = 1
                    else:
                        ret = val[j] / val[j-i]
                    val_ret.append(ret)
                df['{}_change_{}'.format(col, i)] = val_ret

        datetimes = df['datetime'].values
        years, months, days, hours, mins = [],[],[],[],[]
        for d in tqdm(datetimes):
            dtobj = pd.to_datetime(d)
            years.append(dtobj.year)
            months.append(dtobj.month)
            days.append(dtobj.day)
            hours.append(dtobj.hour)
            mins.append(dtobj.minute)
        df['year'], df['month'], df['day'], df['hour'], df['minute'] = years, months, days, hours, mins
        df.dropna(inplace=True)
        df.drop(columns={'datetime',
                         'open',
                         'high',
                         'low',
                         'close',
                         'volume',
                         'vwap',
                         'hwma',
                         'linreg',
                         'year'}, inplace=True)
        return df

    def send_message(self, text):
        try:
            self.telebot.sendMessage(chat_id=self.telegram_chat_id, text=text)
        except Exception as e:
            print(e)

    def set_leverage(self, buy_leverage: int, sell_leverage: int):
        try:
            self.exchange.private_linear_post_position_set_leverage({
                "symbol": self.symbol_id,
                "buy_leverage": buy_leverage,
                "sell_leverage": sell_leverage,
            })
        except ExchangeError:
            # if leverage not changed, bybit throws error thus ignore
            pass

    def position_exists(self):
        positions = self.exchange.private_linear_get_position_list({"symbol": self.symbol_id})["result"]
        buy_position_exists = positions[0]["size"] != '0'
        sell_position_exists = positions[1]["size"] != '0'
        return buy_position_exists or sell_position_exists

    def get_best_bid_ask(self):
        orderbook = self.exchange.fetch_order_book(symbol=self.symbol)
        max_bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
        min_ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
        return max_bid, min_ask

    def place_best_buy_limit_order_bybit_api(self, qty, reduce_only, stop_loss, take_profit):
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

    def place_best_sell_limit_order_bybit_api(self, qty, reduce_only, stop_loss, take_profit):
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

    def my_floor(self, a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    def get_final_high_tp(self, predicted_high):
        if predicted_high >= 0.01 and predicted_high < 0.02:
            return 1.00
        elif predicted_high >= 0.02 and predicted_high < 0.03:
            return 2.00
        elif predicted_high >= 0.03:
            return 3.00
        else:
            return 0.75

    def get_final_low_tp(self, predicted_low):
        if predicted_low <= -0.01 and predicted_low > -0.02:
            return 1.00
        elif predicted_low <= -0.02 and predicted_low > -0.03:
            return 2.00
        elif predicted_low <= -0.03:
            return 3.00
        else:
            return 0.75

    def execute_trade(self):
        # run trade (4 hour cycle: 1 -> 5 -> 9 -> 1 ...)
        iteration = 0
        move = 0 # -1: short, 1: long
        leverage = 1

        # setups needed for scraping news
        self.LM.cuda()
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
                    encoded_inputs = self.tokenizer(titles[i],contents[i],return_tensors="pt",max_length=512,truncation=True).to(self.device)
                    with torch.no_grad():
                        output = self.LM(**encoded_inputs)
                        logits = output['logits']
                        probs = self.softmax(logits)
                        probs = probs.detach().cpu().numpy().flatten()
                        sentiment_scores += probs
            x = df.values[-2].reshape((-1, df.shape[1]))
            sentiment_scores = sentiment_scores[:2].numpy().reshape((1,2))
            x = np.concatenate([x, sentiment_scores], axis=1)
            pred = float(self.tn.predict(x).item())
            prob = np.max(self.tn.predict_proba(x))

            df_hl = self.get_df()
            df_hl = self.create_timestamps(df_hl)
            df_hl = self.preprocess_data_for_HL(df_hl)
            x_hl = df_hl.values[-2].reshape((-1, df_hl.shape[1]))
            predicted_high = self.reg_high.predict(x_hl).item()
            predicted_low = self.reg_low.predict(x_hl).item()
            self.send_message("Predicted High for the next 4 hrs : {:.3f}%".format(predicted_high * 100))
            self.send_message("Predicted Low for the next 4 hrs : {:.3f}%".format(predicted_low * 100))

            pos_dict = {0:'Long', 1:'Short', 2:'Hold'}
            self.send_message("CBITS directional prediction {}, probability {:.3f}".format(pos_dict[pred], prob))
            self.send_message("current BTC news sentiment score (positive, negative): {}".format(sentiment_scores))

            if pred == 0.0:
                if not self.position_exists():
                    self.send_message("currently no positions opened, so we do not need to close any")
                else:
                    if move == -1:
                        self.send_message("closing previous short position and opening long position")
                        self.place_best_buy_limit_order_bybit_api(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)
                    elif move == 1:
                        self.send_message("closing previous short position and opening long position")
                        self.place_best_sell_limit_order_bybit_api(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)
                max_bid, min_ask = self.get_best_bid_ask()
                cur_price = (max_bid + min_ask) / 2.0
                balances = self.exchange.fetch_balance({"coin":"USDT"})["info"]
                usdt = balances["result"]["USDT"]["available_balance"]
                self.send_message("current cash status = " + str(usdt))
                qty = float(usdt) / float(cur_price) * leverage
                qty = self.my_floor(qty, precision=5)
                tp = self.get_final_high_tp(predicted_high)
                stop_loss = float(cur_price) * (1-self.STOP_LOSS_PERCENT/100)
                stop_loss = round(stop_loss)
                take_profit = float(cur_price) * (1+tp/100)
                take_profit = round(take_profit)
                self.place_best_buy_limit_order_bybit_api(
                    qty = qty,
                    reduce_only = False,
                    stop_loss = stop_loss,
                    take_profit = take_profit
                )
                move = 1
            elif pred == 1.0:
                if not self.position_exists():
                    self.send_message("currently no positions opened, so we do not need to close any")
                else:
                    if move == -1:
                        self.send_message("closing previous short position and opening short position")
                        self.place_best_buy_limit_order_bybit_api(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)
                    elif move == 1:
                        self.send_message("closing previous long position and opening short position")
                        self.place_best_sell_limit_order_bybit_api(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)
                max_bid, min_ask = self.get_best_bid_ask()
                cur_price = (max_bid + min_ask) / 2.0
                balances = self.exchange.fetch_balance({"coin":"USDT"})["info"]
                usdt = balances["result"]["USDT"]["available_balance"]
                self.send_message("current cash status = " + str(usdt))
                qty = float(usdt) / float(cur_price) * leverage
                qty = self.my_floor(qty, precision=5)
                tp = self.get_final_low_tp(predicted_low)
                take_profit = float(cur_price) * (1+self.STOP_LOSS_PERCENT/100)
                take_profit = round(take_profit)
                stop_loss = float(cur_price) * (1-tp/100)
                stop_loss = round(stop_loss)
                self.place_best_sell_limit_order_bybit_api(
                    qty = qty,
                    reduce_only=False,
                    stop_loss = take_profit,
                    take_profit = stop_loss
                )
                move = -1
            elif pred == 2.0:
                if not self.position_exists():
                    self.send_message("currently no positions opened, so we do not need to close any")
                else:
                    if move == -1:
                        self.send_message("closing previous short position")
                        self.place_best_buy_limit_order_bybit_api(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)
                    elif move == 1:
                        self.send_message("closing previous long position")
                        self.place_best_sell_limit_order_bybit_api(qty=qty, reduce_only=True, stop_loss=None, take_profit=None)
                if np.abs(predicted_high) >= np.abs(predicted_low):
                    self.send_message("choosing long position")
                    max_bid, min_ask = self.get_best_bid_ask()
                    cur_price = (max_bid + min_ask) / 2.0
                    balances = self.exchange.fetch_balance({"coin":"USDT"})["info"]
                    usdt = balances["result"]["USDT"]["available_balance"]
                    self.send_message("current cash status = " + str(usdt))
                    qty = float(usdt) / float(cur_price) * leverage
                    qty = self.my_floor(qty, precision=5)
                    tp = self.get_final_high_tp(predicted_high)
                    stop_loss = float(cur_price) * (1-self.STOP_LOSS_PERCENT/100)
                    stop_loss = round(stop_loss)
                    take_profit = float(cur_price) * (1+tp/100)
                    take_profit = round(take_profit)
                    self.place_best_buy_limit_order_bybit_api(
                        qty = qty,
                        reduce_only = False,
                        stop_loss = stop_loss,
                        take_profit = take_profit
                    )
                    move = 1
                elif np.abs(predicted_high) < np.abs(predicted_low):
                    self.send_message("choosing long position")
                    max_bid, min_ask = self.get_best_bid_ask()
                    cur_price = (max_bid + min_ask) / 2.0
                    balances = self.exchange.fetch_balance({"coin":"USDT"})["info"]
                    usdt = balances["result"]["USDT"]["available_balance"]
                    self.send_message("current cash status = " + str(usdt))
                    qty = float(usdt) / float(cur_price) * leverage
                    qty = self.my_floor(qty, precision=5)
                    tp = self.get_final_low_tp(predicted_low)
                    take_profit = float(cur_price) * (1+self.STOP_LOSS_PERCENT/100)
                    take_profit = round(take_profit)
                    stop_loss = float(cur_price) * (1-tp/100)
                    stop_loss = round(stop_loss)
                    self.place_best_sell_limit_order_bybit_api(
                        qty = qty,
                        reduce_only = False,
                        stop_loss = take_profit,
                        take_profit = stop_loss
                    )
                    move = -1

            iteration += 1
            self.send_message("waiting for the next 4 hours \(^.^)/")
            elapsed = time.time() - t0
            time.sleep(60 * 60 * 4 - elapsed)


if __name__ == '__main__':
    bybit_cred = {
        "api_key": "wQEux6LxRKuEFx6U3M",
        "api_secret": "NrIWVue3JH4L2c0rwowRCVPJaQ59LcQIY03y",
    }
    telegram_cred = {
        "token": "5322673870:AAHO3hju4JRjzltkG5ywAwhjaPS2_7HFP0g",
        "chat_id": 1720119057,
    }
    trader = CBITS_v3(
        symbol="BTCUSDT",
        bybit_credential=bybit_cred,
        telegram_credential=telegram_cred,
        cbits_zip_dir="tabnet_roberta_full.zip",
        tabnet_high_dir="reg_high_chart_only_seed0.zip",
        tabnet_low_dir="reg_low_chart_only_seed0.zip"
    )


trader.execute_trade()
