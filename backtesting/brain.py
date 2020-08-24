import pandas as pd
import time
from datetime import datetime
import pickle
import numpy as np

import statistics as s
import heart as h

class Brain():
    SPECIAL_NAME='inprogress'

    TARGET = 0.01 
    '''
    LEVERAGE = 5
    ORDER_SIZE = 0.1
    PYRAMID_MAX = 2

    THRESHOLD = 0.1 #zakres 0-0.5
    '''

    MODEL = '15m-ehh-24.08.20/BTCUSDT15m-100x50~0.01-24Aug20-23.26.47/05-TL0.690-TA0.533_VL0.687-VA0.533.model'
    #secondary_model = '15m-new-10.08.20/BTCUSDT15m-100x50~0.01-11Aug20-12.33.21/08-TL0.686-TA0.544_VL0.687-VA0.558.model'

    comment = "testing testing ml simple"



    def decision(self, heart, pred, change, price):
        if change>=(self.TARGET) or change<=-(self.TARGET):
            heart.close("SHORT")
            heart.close("LONG")
        
        if (pred>0.5+self.THRESHOLD): # and (pred_2<0.5+self.THRESHOLD):
            heart.close("SHORT")
           # if (pred_2<0.5+self.THRESHOLD):
            heart.create_order("LONG", price*0.999) 

        elif (pred<0.5-self.THRESHOLD): # and (pred_2<0.5+self.THRESHOLD):
            heart.close("LONG")
            #if (pred_2<0.5+self.THRESHOLD):
            heart.create_order("SHORT", price*1.001)

    def loop(self, heart):
        while True:
            candles, pred = heart.tick()

            if candles.empty:
                heart.close("SHORT", cancel_awaiting_orders=True)
                heart.close("LONG", cancel_awaiting_orders=True)
                return 0

            #print("Candles ", datetime.fromtimestamp(candles[0].iloc[-1]/1000))

            #heart.print_details()

            self.decision(heart, pred, heart.change, heart.price)

            #time.sleep(3)

    def test_many(self):
        thresholds = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16]
        leverages = [7]
        order_sizes = [0.1]
        pyramid_maxes = [1]


        for t in thresholds:
            for l in leverages:
                for o in order_sizes:
                    for p in pyramid_maxes:
                        self.test(self.MODEL, t, l, o, p)


    def test(self, MODEL, THRESHOLD, LEVERAGE, ORDER_SIZE, PYRAMID_MAX):
        self.MODEL = MODEL
        self.THRESHOLD = THRESHOLD
        self.LEVERAGE = LEVERAGE
        self.ORDER_SIZE = ORDER_SIZE
        self.PYRAMID_MAX = PYRAMID_MAX

        heart = h.Account(MODEL=self.MODEL,
                    #secondary_model = self.secondary_model,
                    leverage=self.LEVERAGE, 
                    order_size=self.ORDER_SIZE, 
                    pyramid_max=self.PYRAMID_MAX, 
                    FEE=0.0005, 
                    liq_bump=0.004, 
                    start_balance=100)

        name = f"{heart.symbol}{heart.interval}-T{self.THRESHOLD}-L{self.LEVERAGE}O{self.ORDER_SIZE}P{self.PYRAMID_MAX}-{self.SPECIAL_NAME}-{datetime.now().strftime('%d%b%y-%H.%M.%S')}"
        print(name)

        self.loop(heart)
        #s.get_trades()
        s.save_trades(name, self.MODEL, self.comment)   


b = Brain()
b.test_many()