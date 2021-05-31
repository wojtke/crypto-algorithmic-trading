from utils import create_dir, date_to_ms, date_from_ms, now
import pandas as pd
from datetime import datetime
from binance.client import Client
from time import sleep

def download_data():
    intervals = ['4h', '1d']
    symbols = ['BTCUSDT']
    market = 'spot'

    for interval in intervals:
        for symbol in symbols:

            if market == 'spot':
                filename = f"RAW_DATA/bBinance_{symbol}_{interval}.json"
            elif market == 'futures':
                filename = f"RAW_DATA/Binance_futures_{symbol}_{interval}.json"

            try:
                klines = pd.read_json(filename)
                start_ts = klines[0].iloc[-1]+1
                print(f"Data for {symbol}{interval} found in local files: {len(klines)}")
            except ValueError:
                print(f"No data found for {symbol}{interval}")
                start_ts = None
                klines = pd.DataFrame()

            print("--------")
            print(f"Time right now: {now()}")
            print(f"Symbol: {symbol}, interval: {interval}")
            if start_ts:
                print(f"Download starting from: {date_from_ms(start_ts)}")
            else:
                print(f"Download starting from the beggining")

            new_klines = get_historical_klines(symbol, interval, start_ts=start_ts, market=market)

            print(f"New klines: {len(new_klines)}")
            klines = klines.append(new_klines, ignore_index=True)
            print(f"Total klines: {len(klines)}")

            print('Saving...', end=" ")
            klines.to_json(filename)
            print('done')


def get_historical_klines(symbol, interval, start_ts=None, market = 'spot'):
    client = Client("","")
    limit = 500

    if not start_ts:
        start_ts = date_to_ms("1 Sep, 2017")

    if market == 'spot':
        get_function = client.get_klines
    elif market=='futures':
        get_function = client.futures_klines

    output_data = []
    i=0

    while True:
        if i%10==0:
            sleep(0.5)
        i+=1

        temp_data = get_function(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=None
        )

        output_data += temp_data

        if len(temp_data) < limit:
            break

        start_ts = temp_data[-1][0] + 1

    return output_data


if __name__ == '__main__':
    download_data()