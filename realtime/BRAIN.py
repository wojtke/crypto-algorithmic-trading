import pandas as pd
import time
from datetime import datetime
import pickle
from tensorflow import keras
import numpy as np

from GENERAL import General
from ORDERS import Orders

from sys import path
path.append("..")
from vars import Vars

class Brain:
    def __init__(self):
        self.SYMBOL = "BTC"
        self.INTERVAL = '15m'
        self.LEVERAGE = 1
        self.ORDER_SIZE = 0.1
        self.PYRAMID_MAX = 1

        self.PAST_KLINES = 100
        self.TARGET = 0.01

        self.THRESHOLD = 0.06 #zakres 0-0.5

        self.MODEL = '15m-new-10.08.20/BTCUSDT15m-100x50~0.01-11Aug20-12.33.21/08-TL0.686-TA0.544_VL0.687-VA0.558.model'

        self.MODEL_PATH = Vars.main_path + "MODELS/" + self.MODEL
        self.SCALER_PATH = Vars.main_path + "SCALERS/" + self.MODEL[:-58] + '-scaler_data.pickle'

        self.general = General(is_it_for_real=True)
        self.orders = Orders(self.SYMBOL,
                            self.LEVERAGE,
                            self.PYRAMID_MAX,
                            self.ORDER_SIZE,
                            is_it_for_real=True)

        self.pickle_scaler_data = pickle.load(open( self.SCALER_PATH, "rb" ))
        print("SCALER: ", self.pickle_scaler_data)

        self.model = keras.models.load_model(self.MODEL_PATH)

    def decision(self, pred, price):
        if (pred>0.5+self.THRESHOLD):
            print("LONG")
            pass
            #self.orders.close("SHORT")
            #self.orders.create_order("LONG", price*0.999) 

        elif (pred<0.5-self.THRESHOLD):
            print("SHORT")
            pass
            #self.orders.close("LONG")
            #self.orders.create_order("SHORT", price+*1.001)

    def loop(self):
        while True: 
            print("Start")
            loopy=1
            while loopy !=0:
                time.sleep(180) 

                loopy = self.general.wait_till(self.INTERVAL, advance=10, skip_till=200)

                t = time.time()
                self.orders.update()
                print(time.time()-t, "seconds for update")

                #self.candles_get()
        

            candles, candles_future, price, price_future= self.candles_get()

            pred = self.get_prediction(candles)
            pred_future = self.get_prediction(candles_future)

            print(f"Prediction: {pred}, futures pred: {pred_future}")

            self.decision(pred, price)


    def candles_get(self):
        candles = self.general.get_candles(self.SYMBOL, self.INTERVAL, self.PAST_KLINES)
        candles_future = self.general.get_candles(self.SYMBOL, self.INTERVAL, self.PAST_KLINES, market='future')

        price = float(candles.values[-1][4])
        price_future = float(candles_future.values[-1][4])

        print(f"Price: {price}, futures price: {price_future}")

        print("Candles ", datetime.fromtimestamp(candles[0].iloc[-1]/1000))
        print(candles.tail(5))

        return candles, candles_future, price, price_future

    def get_prediction(self, candles):
        candles.columns = ['openTimestamp', 'open', 'high', 'low', 'close', 'volume']

        for c in candles.columns:
            candles[c] = pd.to_numeric(candles[c], errors='coerce')

        candles.dropna(subset=['close', 'volume', 'high', 'low'], inplace=True)

        candles['low'] = candles['close']/candles['low'] #low wick
        candles['high'] = candles['high']/candles['close'] #high wick

        candles['close'] = candles['close'].pct_change()
        candles['volume'] = candles['volume'].add(1) #to jest po to, zeby nie wywalac swiec o volume 0, bo wtedy pct_change -> inf
        candles['volume'] = candles['volume'].pct_change() 

        candles.drop(columns=['open', 'openTimestamp'], inplace=True)  #drop unused columns

        candles.dropna(subset=['close', 'volume', 'high', 'low'], inplace=True)

        for i, col in enumerate(candles.columns): 
            candles[col] = candles[col].sub(self.pickle_scaler_data[0][i]).div(self.pickle_scaler_data[1][i])
        
        prediction = self.model.predict([np.array([np.array(candles.values)])])

        return prediction[0][1]


brain = Brain()
brain.loop()