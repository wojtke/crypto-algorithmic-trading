import pandas as pd
import time
from datetime import datetime
import pickle
import numpy as np

import statistics as s
import heart as h

SPECIAL_NAME='inprogress'

SYMBOL = "BTC"
INTERVAL = '15m'
LEVERAGE = 30
ORDER_SIZE = 0.1
PYRAMID_MAX = 1

PAST_KLINES = 100
TARGET = 0.01

THRESHOLD = 0.1 #zakres 0-0.5

MODEL = '15m-normal-11.07.20/BTCUSDT15m-100x50~0.01-normal-11Jul20-17.59.26/07-TL0.679-TA0.564_VL0.687-VA0.563.model'

name = f"{SYMBOL}{INTERVAL}-T{THRESHOLD}-L{LEVERAGE}O{ORDER_SIZE}P{PYRAMID_MAX}-{SPECIAL_NAME}-{datetime.now().strftime('%d%b%y-%H.%M.%S')}"
comment = "testing testing ml simple"


def decision(pred, change, price):
    
    if change>=TARGET or change<=-TARGET:
        heart.close("SHORT")
        heart.close("LONG")
    
    if (pred>0.5+THRESHOLD):
        heart.close("SHORT")
        heart.create_order("LONG", price-5) 

    elif (pred<0.5-THRESHOLD):
        heart.close("LONG")
        heart.create_order("SHORT", price+5)

def loop():
    while True:
        candles, pred = heart.tick()

        if candles.empty:
            heart.close("SHORT", cancel_awaiting_orders=True)
            heart.close("LONG", cancel_awaiting_orders=True)
            return 0

        #print("Candles ", datetime.fromtimestamp(candles[0].iloc[-1]/1000))

       # heart.print_details()

        decision(pred, heart.change, heart.price)

        #time.sleep(3)

def test_many():
    global INTERVAL, LEVERAGE, RSI_PERIOD, PAST_KLINES, name
    intervals = ['15m', '5m']
    leverages = [1]
    rsi_periods = [5,7,9,12,15,20]
    PAST_KLINESs = [300,250,200,150,100,50,25]

    iv_sec = {'1m':60, '3m':60*3, '5m':60*5, '15m':60*15, '30m':60*30, '1h':60*60, '2h':60*60*2, '4h':60*60*4, '8h':60*60*8, '1d':60*60*24, '3d':60*60*24*3, '1w':60*60*24*7}
    a=0
    for i in intervals:
        a+=110/iv_sec[i]
    print("Estimated time to test all: ", round(a*len(PAST_KLINESs)*len(rsi_periods)*len(leverages),2) , " hours")

    for i in intervals:
        for l in leverages:
            for r in rsi_periods:
                for b in PAST_KLINESs:
                    INTERVAL = i
                    LEVERAGE = l
                    RSI_PERIOD = r
                    PAST_KLINES = b

                    name = f"{SYMBOL}{INTERVAL}-Lev{LEVERAGE}Ord{ORDER_SIZE}Pyr{PYRAMID_MAX}-Boll{PAST_KLINES}Rsi{RSI_PERIOD}-{round(time.time())}"
                    print("Interval", i,
                        "\nLeverage", l,
                        "\nRsi period", r,
                        "\nBoll period", b)

                    test()

def test():
    loop()
    s.get_trades()
    s.save_trades(name, MODEL, comment)   



heart = h.Account(symbol=SYMBOL, 
                interval = INTERVAL, 
                leverage=LEVERAGE, 
                order_size=ORDER_SIZE, 
                pyramid_max=PYRAMID_MAX, 
                FEE=0.0000, 
                liq_bump=0.004, 
                start_balance=100, 
                kline_limit=PAST_KLINES+1,
                MODEL=MODEL)


test()