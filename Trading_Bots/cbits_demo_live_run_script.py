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
import requests
from dateutil import parser
from binance.client import Client
from binance.helpers import round_step_size



class CBITS_Binance_v1:
    stop_percent = 2.0
    target_percent = 0.75

    def __init__(self, binance_credential, telegram_credential, cbits_ckpt, tabnet_direction_ckpt):
        # ccxt binance object for getting dataframe and possibly other useful information
        self.exchange: BinanceExchange = ccxt.binance({
            "enableRateLimit": True,
            "apiKey": binance_credential["api_key"],
            "secret": binance_credential["api_secret"]
        })

        # python-binance object
        self.client = Client(api_key=binance_credential["api_key"],
                             api_secret=binance_credential["api_secret"])

        self.trader = TabNetClassifier()
        self.trader.load_model(cbits_ckpt)

        self.directional_prophet = TabNetClassifier()
        self.directional_prophet.load_model(tabnet_direction_ckpt)

        self.telebot = telegram.Bot(token=telegram_credential["token"])
        self.telegram_chat_id = telegram_credential["chat_id"]

        self.symbol = "BTCUSDT"

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
    def preprocess_data_for_TabNet_Traders(df):
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
        df['bop'] = df.ta.bop(lookahead=False)
        df['ebsw'] = df.ta.ebsw(lookahead=False)
        df['cmf'] = df.ta.cmf(lookahead=False)
        df['vwap'] = df.ta.vwap(lookahead=False)

        df['rsi/100'] = df.ta.rsi(lookahead=False) / 100
        df['rsx/100'] = df.ta.rsx(lookahead=False) / 100

        df['high/low'] = df['high'] / df['low']
        df['close/open'] = df['close'] / df['open']
        df['high/open'] = df['high'] / df['open']
        df['low/open'] = df['low'] / df['open']

        df['hwma'] = df.ta.hwma(lookahead=False)
        df['linreg'] = df.ta.linreg(lookahead=False)
        df['hwma/close'] = df['hwma'] / df['close']
        df['linreg/close'] = df['linreg'] / df['close']

        df['ema_10'] = df.ta.ema(length=10, lookahead=False)
        df['ema_60'] = df.ta.ema(length=60, lookahead=False)
        df['ema_120'] = df.ta.ema(length=120, lookahead=False)

        # differencing
        for l in range(1,12):
            for col in ['open','high','low','close','volume', 'vwap', 'ema_10', 'ema_60', 'ema_120']:
                val = df[col].values
                val_ret = [None for _ in range(l)]
                for i in range(l, len(val)):
                    if val[i-l] == 0:
                        ret = 1
                    else:
                        ret = val[i] / val[i-l]
                    val_ret.append(ret)
                df['{}_change_{}'.format(col, l)] = val_ret

        df = df.dropna()
        close_prices = df['close'].values
        df = df.drop(columns={'year',
                              'datetime',
                              'open',
                              'high',
                              'low',
                              'close',
                              'volume',
                              'vwap',
                              'ema_10',
                              'ema_60',
                              'ema_120'})
        return df, close_prices

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
            symbol="BTCUSDT",
            side="BUY",
            quantity=qty,
            reduceOnly=reduce_only,
            type="MARKET")

        if reduce_only == False: # send in stop loss and take profit
            futures_stop_loss = self.client.futures_create_order(
                symbol="BTCUSDT",
                timeInForce="GTC",
                side="SELL",
                type="STOP_MARKET",
                stopPrice=stopPrice,
                closePosition=True)

            futures_take_profit = self.client.futures_create_order(
                symbol="BTCUSDT",
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
            symbol="BTCUSDT",
            side="SELL",
            quantity=qty,
            reduceOnly=reduce_only,
            type="MARKET")

        if reduce_only == False: # send in take profit and stop loss
            futures_stop_loss = self.client.futures_create_order(
                symbol="BTCUSDT",
                timeInForce="GTC",
                side="BUY",
                type="STOP_MARKET",
                stopPrice=str(stopPrice),
                closePosition=True)

            futures_take_profit = self.client.futures_create_order(
                symbol="BTCUSDT",
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

    def execute_trade(self):
        # run trade (4 hour cycle: 1 -> 5 -> 9 -> 1 ...) KST
        iteration = 0
        move = 0 # -1: short, 1: long
        stoploss_id, takeprofit_id = -1, -1 # stop loss and take profit id, close them after a single iteration just in case they are not filled
        while True:
            self.send_message("========== Iteration {} ==========".format(iteration))
            t0 = time.time()
            df = self.get_df()
            df = self.create_timestamps(df)

            df, close = self.preprocess_data_for_TabNet_Traders(df)
            x = df.values[-2].reshape((-1, df.shape[1]))
            pred = float(self.trader.predict(x).item())
            prob = np.max(self.trader.predict_proba(x))

            prev_close = close[-2]

            pred = 0.0
            pos_dict = {0:'Long', 1:'Short', 2:'Hold'}
            self.send_message("CBITS directional prediction {}, probability {:.3f}".format(pos_dict[pred], prob))

            if iteration > 0:
                # get rid of unclosed take profit and stop loss orders
                try:
                    self.client.futures_cancel_order(symbol="BTCUSDT", orderId=stoploss_id)
                except Exception as e:
                    print(e)
                try:
                    self.client.futures_cancel_order(symbol="BTCUSDT", orderId=takeprofit_id)
                except Exception as e:
                    print(e)
                # close if there are any positions open from the previous iteration
                qty = self.get_position_size()
                if qty == 0:
                    self.send_message("no positions open... stop loss or take profit was probably triggered.")
                else:
                    if move == -1:
                        self.send_message("Closing previous short position...")
                        self.place_best_buy_limit_order(reduce_only=True, qty=qty, stopPrice=None, targetPrice=None)
                    elif move == 1:
                        self.send_message("Closing previous long position...")
                        self.place_best_sell_limit_order(reduce_only=True, qty=qty, stopPrice=None, targetPrice=None)


            if pred == 0.0: # long
                btc_usdt = self.client.get_symbol_ticker(symbol="BTCUSDT")
                btc_usdt = float(btc_usdt['price'])

                stopPrice = prev_close * (1 - self.stop_percent/100)
                targetPrice = prev_close * (1 + self.target_percent/100)

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
            elif pred == 1.0:
                btc_usdt = self.client.get_symbol_ticker(symbol="BTCUSDT")
                btc_usdt = float(btc_usdt['price'])

                stopPrice = prev_close * (1 + self.stop_percent/100)
                targetPrice = prev_close * (1 - self.target_percent/100)

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
            elif pred == 2.0:
                self.send_message("CBITS chose hold, but we will still trade based on our directional prophet.")
                directional_pred = self.directional_prophet.predict(x).item()
                if directional_pred == 0.0:
                    btc_usdt = self.client.get_symbol_ticker(symbol="BTCUSDT")
                    btc_usdt = float(btc_usdt['price'])

                    stopPrice = prev_close * (1 - self.stop_percent/100)
                    targetPrice = prev_close * (1 + self.target_percent/100)

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
                else:
                    btc_usdt = self.client.get_symbol_ticker(symbol="BTCUSDT")
                    btc_usdt = float(btc_usdt['price'])

                    stopPrice = prev_close * (1 + self.stop_percent/100)
                    targetPrice = prev_close * (1 - self.target_percent/100)

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


binance_cred = {
    "api_key": "<BINANCE API KEY>",
    "api_secret": "<BINANCE SECRET KEY>",
}
telegram_cred = {
    "token": "<TELEGRAM TOKEN>",
    "chat_id": <TELEGRAM CHAT ID>,
}
trader = CBITS_Binance_v1(
    binance_credential=binance_cred,
    telegram_credential=telegram_cred,
    cbits_ckpt="tabnet_clf_chart_only_v2.zip",
    tabnet_direction_ckpt="TabNet_high_low_vol.zip"
)

trader.execute_trade()
