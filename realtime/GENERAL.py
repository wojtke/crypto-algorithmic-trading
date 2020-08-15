from datetime import datetime
import time
import ta
import pandas as pd

class General:
    def __init__(self, is_it_for_real=False):
        self.iv_sec = {'1m':60, '3m':60*3, '5m':60*5, '15m':60*15, '30m':60*30, '1h':60*60, '2h':60*60*2, '4h':60*60*4, '8h':60*60*8, '1d':60*60*24, '3d':60*60*24*3, '1w':60*60*24*7}

        if is_it_for_real:
            from binance.client import Client
            self.client = Client("", "")

    def wait_till(self, interval, advance=0, skip_till=None):
        lag = self.client.get_server_time()['serverTime']/1000 - time.time() + 0.5 #taka kalibracja, żeby serwer otrzymał nasz request ułamek sekundy przed rządaną przez nas chwilą
        print('Lag: ', round(lag,3))

        to_wait = (-time.time()-advance-lag)%self.iv_sec[interval]
        print(f"Candle ({interval}) closing in {round(to_wait,3)} seconds")
        if skip_till:
            if to_wait>skip_till:
                print("Skipping the wait")
                return to_wait

        time.sleep(to_wait)
        print(f"Server time: {datetime.fromtimestamp(self.client.get_server_time()['serverTime']/1000)}")
        return 0

    def get_candles(self, symbol, interval, limit, market='spot'):
        if market == 'spot':
            candles = pd.DataFrame(self.client.get_klines(symbol=symbol+'USDT', interval=interval, limit=limit+1))
        elif market == 'future' or market == 'futures':
            candles = pd.DataFrame(self.client.futures_klines(symbol=symbol+'USDT', interval=interval, limit=limit+1))

        candles = candles[[0,1,2,3,4,5]].astype(float)

        return candles

