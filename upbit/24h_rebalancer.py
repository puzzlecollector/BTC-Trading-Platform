import numpy as np 
import pandas as pd 
import pyupbit
import time 
import datetime 
import random 
import os 
import pickle 
import telegram 
import asyncio
import requests

telebot = telegram.Bot(token="5794337270:AAFre732XikwrYtahljvFiIPwq5nlvQCfW4") 
tele_token = "5794337270:AAFre732XikwrYtahljvFiIPwq5nlvQCfW4" 
chat_id = 1720119057 
access = "GVmdifAVnFTBFEh0P0vlndx1rwisxdaE0n5Y4X7I" 
secret = "S8WwEavEz5hze2sxv3seP4pUKx7rncgMx7ixInTB" 
upbit = pyupbit.Upbit(access, secret)  


def send_message(msg):
    url = f"https://api.telegram.org/bot{tele_token}/sendMessage?chat_id={chat_id}&text={msg}"
    print(requests.get(url).json()) 


def GetTopCoinList(interval, top):
    print("----- GetTopCoinList Start -----") 
    Tickers = pyupbit.get_tickers("KRW") 
    time.sleep(0.1) 
    dict_coin_money = dict() 
    for ticker in Tickers:
        print("-"*30, ticker) 
        try:
            time.sleep(0.1) 
            df = pyupbit.get_ohlcv(ticker, interval) 
            volume_money = df["value"][-2] + df["value"][-1] 
            dict_coin_money[ticker] = volume_money 
            print(ticker, dict_coin_money[ticker]) 
        except Exception as e:
            print(e) 

    dict_sorted_coin_money = sorted(dict_coin_money.items(), key=lambda x : x[1], reverse=True) 
    coin_list = list() 
    cnt = 0 
    for coin_data in dict_sorted_coin_money:
        cnt += 1 
        if cnt <= top:
            coin_list.append(coin_data[0]) 
        else:
            break 
    print("----- GetTopCoinList End -----") 
    return coin_list 

iterations = 0 

while True: 
    send_message(f"===== Iteration {iterations} =====")
    if iterations > 0:
        # close positions 
        for i in range(len(topList)):
            unit = upbit.get_balance(topList[i]) 
            upbit.sell_market_order(topList[i], unit) 
            
    topList = GetTopCoinList("day", 10)   
    send_message(f"Investing in : {str(topList)}") 
    cash_amount = upbit.get_balance("KRW") 
    send_message(f"Current cash amount : {cash_amount} KRW") 
    portfolio_weights = [1/10 for _ in range(10)] 
    gamma = 0.05 
    for i in range(len(topList)):
        coin_name = topList[i] 
        order_amount = cash_amount * (1 - gamma/100) * portfolio_weights[i] 
        upbit.buy_market_order(coin_name, order_amount) 
        time.sleep(0.1) 

    iterations += 1 
    send_message(f"Waiting for the next 24 hours") 
    time.sleep(60*60*24) 
