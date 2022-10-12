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

class CBITS_S1:
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

    
