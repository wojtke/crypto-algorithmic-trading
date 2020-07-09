import time
import dateparser
import pytz
import json
import pandas as pd
import os


from datetime import datetime
from binance.client import Client

#start = "1 Sep, 2017"
#end = "25 Mar, 2020"

symbols = ['BTCUSDT', 'ETHUSDT', 'ETHBTC']
intervals = ['1m','5m','15m','1h', '4h', '1d']

def update_all():

    try:#
        os.makedirs(f"RAW_DATA")

    except  FileExistsError:
        pass

    for interval in intervals:
        for symbol in symbols:

            filename = f"RAW_DATA/Binance_{symbol}_{interval}.json"
            print(filename)

            try:
                klines = pd.read_json(filename)
                print("Bylo: ",len(klines))

                print("----- Robie od ", datetime.fromtimestamp(klines[0].iloc[-1]/1000) , symbol, interval, "Teraz jest", datetime.now(), "-----")
                new_klines = get_historical_klines(symbol, interval, start_ts=(klines[0].iloc[-1]+1))
                print("Nowych: ",len(new_klines))
                klines = klines.append(new_klines, ignore_index=True)
                print("Jest: ",len(klines))

                print('zapisujeee')
                klines.to_json(filename)
            except:
                print("----- Robie od początku ", symbol, interval, datetime.now(), "-----")
                klines = get_historical_klines(symbol, interval)
                print('zapisujeee')
                with open(filename, 'w') as f:
                    f.write(json.dumps(klines))

def specific():
    create_dir("RAW_DATA")
    
    for interval in intervals:
        for symbol in symbols:
            print("----- Startin ", symbol, interval, datetime.now(), "-----")
            klines = get_historical_klines(symbol, interval, start, end)
            print('zapisujeee')
            with open(
                "RAW_DATA/Binance_{}_{}_{}-{}.json".format(
                    symbol,
                    interval,
                    start,
                    end
                 ),
                'w'  # set file write mode
            ) as f:
                f.write(json.dumps(klines))

def create_dir(path):
    try:  
        os.mkdir(path)  
    except OSError as error:  
        print(error)  
def date_to_milliseconds(date_str):
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

def interval_to_milliseconds(interval):
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms

def get_historical_klines(symbol, interval, start_str=None, start_ts=None, end_str=None):
    # create the Binance client, no need for api key
    client = Client("","")

    # init our list
    output_data = []

    # setup the max limit
    limit = 500

    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)

    # convert our date strings to milliseconds
    if start_str:
        start_ts = date_to_milliseconds(start_str)
    elif not start_ts:
        start_ts = date_to_milliseconds("1 Sep, 2017")

    # if an end time was passed convert it
    end_ts = None
    if end_str:
        end_ts = date_to_milliseconds(end_str)

    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if set
        temp_data = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts
        )

        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_data):
            symbol_existed = True

        if symbol_existed:
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe
            if idx % 10 == 0:
                time.sleep(1)

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 15 == 0:
            # time.sleep(1)
            print(f'Robie teraz {symbol} {interval}', datetime.fromtimestamp(start_ts/1000))

    return output_data


update_all()
