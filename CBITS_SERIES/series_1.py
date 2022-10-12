import numpy as np
import pandas as pd
import pandas_ta as ta
import telegram
from ccxt.base.errors import ExchangeError
from ccxt.bybit import bybit as BybitExchange
from pytorch_tabnet.tab_model import TabNetclassifier, TabNetRegressor
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from transformers import XLMRobertaForSequenceClassification, AutoModel
from tokenization_roberta_spm import FairSeqRobertaSentencePieceTokenizer
import requests
from bs4 import BeautifulSoup, Comment
from dateutil import parser
from pybit import HTTP
import pprint

class CBITS_PERFORMANTE:
    STOP_LOSS_PERCENT = 1.5
    def __init__(self, symbol, bybit_credential, telegram_credential, model_chkpt):
        self.exchange: BybitExchange = ccxt.bybit({
            "enableRateLimit": True,
            "apiKey": bybit_credential["api_key"],
            "secret": bybit_credential["api_secret"]
        })
        self.exchange.load_markets()
        self.symbol = symbol
        self.symbol_id = self.exchange.market(self.symbol)["id"]

        ## some loading code model

        self.telebot = telegram.Bot(token=telegram_credential["token"])
        self.telegram_chat_id = telegram_credential["chat_id"]

        self.tokenizer = FairSeqRobertaSentencePieceTokenizer.from_pretrained("fairseq-roberta-all-model")
        self.LM = XLMRobertaForSequenceClassification.from_pretrained("totoro4007/cryptoroberta-base-finetuned")
        self.softmax(dim=1)

        self.session = HTTP(endpoint = "https://api.bybit.com",
                            api_key = bybit_credential["api_key"],
                            api_secret = bybit_credential["api_secret"],
                            spot = False)

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
