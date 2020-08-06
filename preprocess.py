import pandas as pd
from utils import Progressbar, Logger
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
from sklearn import preprocessing
import pickle
from collections import deque
import random
import numpy as np
from time import time
import os
import ta

SYMBOL = 'BTCUSDT'

SPECIAL_NAME = 'normal'

iv_sec = {'1m': 60, '3m': 60 * 3, '5m': 60 * 5,
          '15m': 60 * 15, '30m': 60 * 30, '1h': 60 * 60,
          '2h': 60 * 60 * 2, '4h': 60 * 60 * 4, '8h': 60 * 60 * 8,
          '1d': 60 * 60 * 24, '3d': 60 * 60 * 24 * 3, '1w': 60 * 60 * 24 * 7}

RAW_DATA_PATH = 'D:/PROJEKTY/Python/BINANCE_RAW_DATA'

INTERVAL = '15m'

pasts = [150]
futures = [75]
pcts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2]
pcts = [1]


def load_raw_data():
    print("Loading data...")
    global df
    raw_data_path = f"{RAW_DATA_PATH}/Binance_{SYMBOL}_{INTERVAL}.json"
    df = pd.read_json(raw_data_path)
    df = df[[0, 1, 2, 3, 4, 5]]
    df.columns = ['openTimestamp', 'open', 'high', 'low', 'close', 'volume']

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # df = df.reset_index()
    # df = df.drop(columns=["index"])

    print("...loading done\n")
    return df


def classify(df):
    print("Classifying data...")

    leng = int(len(df.index) - FUTURE_CHECK_LEN) - 1

    counter = {"H": 0, "L": 0, "N": 0, "B": 0}
    classification = []

    pb = Progressbar(leng, name="Classification")

    if SPECIAL_NAME == 'normal':  # NORMALNE LICZENIE TARGETU JAKO PCT CHANGE
        for i in range(leng):  # mozna zoptymalizowac troche
            high, low = classification_target(i)

            counter_case, to_append = assign_classification(high, low, i + 1)
            classification.append(to_append)
            counter[counter_case] += 1

            pb.update(i)

    elif SPECIAL_NAME == 'stdev':  # LICZENIE TARGETU PROPORCJONALNIE DO STDEV
        df['stdev'] = ta.volatility.bollinger_hband(close=df['close'], n=PAST_SEQ_LEN,
                                                    ndev=0.5) - ta.volatility.bollinger_lband(close=df['close'],
                                                                                              n=PAST_SEQ_LEN, ndev=0.5)
        for i in range(leng):
            high = df["close"][i] + TARGET_CHANGE * df['stdev'][i]
            low = df["close"][i] - TARGET_CHANGE * df['stdev'][i]

            counter_case, to_append = assign_classification(high, low, i + 1)
            classification.append(to_append)
            counter[counter_case] += 1

            pb.update(i)
        df.drop(columns=['stdev'], inplace=True)

    elif SPECIAL_NAME == 'stdevcut':
        df['stdev'] = ta.volatility.bollinger_hband(close=df['close'], n=PAST_SEQ_LEN, ndev=0.5) \
                      - ta.volatility.bollinger_lband(close=df['close'], n=PAST_SEQ_LEN, ndev=0.5)

        for i in range(leng):
            if df['stdev'][i] / df["close"][i] >= TARGET_CHANGE:
                high = df["close"][i] + df['stdev'][i]
                low = df["close"][i] - df['stdev'][i]

                counter_case, to_append = assign_classification(high, low, i + 1)
                classification.append(to_append)
                counter[counter_case] += 1
            else:
                classification.append(None)
                counter["N"] += 1

            pb.update(i)
        df.drop(columns=['stdev'], inplace=True)
    else:
        raise Exception("Special name not defined for classification")

    del pb

    print(counter)
    log.log(counter)

    classification = pd.DataFrame({'target': classification})
    df = pd.concat((df, classification), axis=1)

    print("...classifying done\n")
    return df


def classification_target(i):
    high = (1 + TARGET_CHANGE) * df['close'][i]
    low = (1 - TARGET_CHANGE) * df['close'][i]

    return high, low


'''
    high = 
    low =
'''


def assign_classification(high, low, i):  #
    for j in range(FUTURE_CHECK_LEN):
        is_high = df['high'][i + j] >= high
        is_low = df['low'][i + j] <= low

        if is_high:
            if is_low:
                return ("B", None)
            else:
                return ("H", 1)
        elif is_low:
            return ("L", 0)
    return ("N", None)


def to_pct(df):
    print("Changing values to pct change...")

    # to pct
    pre_dropna = len(df.index)
    df.dropna(subset=['close', 'volume', 'high', 'low'], inplace=True)

    df['low'] = df['close'] / df['low']  # low wick
    df['high'] = df['high'] / df['close']  # high wick

    df['close'] = df['close'].pct_change()
    df['volume'] = df['volume'].add(1)  # to jest po to, zeby nie wywalac swiec o volume 0, bo wtedy pct_change -> inf
    df['volume'] = df['volume'].pct_change()

    df.drop(columns=['open'], inplace=True)  # drop unused columns except openTimestamp

    df.dropna(subset=['close', 'volume', 'high', 'low'], inplace=True)
    print(f"Dropped {pre_dropna - len(df.index)} rows")

    print("...changing to pct done\n")
    return df


def separate(df):
    print("Separating data...")

    df_t = pd.DataFrame()
    df_v = pd.DataFrame(data={
        "openTimestamp": [1337]})  # this is passed just for this df to have an index of 0, this row is later dropped

    # too small timedelta on smaller intervals can cause problems

    '''first try but i think its best to use freshest data possible
    #28 jan 2019 low volatility down ts 1548633600000
    #16 mar 2019 low volatility up  1552694400000
    #17 may 2019 high volatility up 1558051200000
    #14 aug 2019 high volatility down ts 1565740800000
    windows = [ [parse_datetime("2019"+"01"+"28"), timedelta(days=8)],
                [parse_datetime("2019"+"03"+"16"), timedelta(days=8)],
                [parse_datetime("2019"+"05"+"16"), timedelta(days=8)],
                [parse_datetime("2019"+"08"+"15"), timedelta(days=8)] ] 
    '''

    windows = [[parse_datetime("2020" + "02" + "23"), timedelta(days=15)],
               [parse_datetime("2020" + "04" + "10"), timedelta(days=15)], ]

    end = df.iloc[-1]['openTimestamp']

    for window in windows:
        ixs = index_window(end, window[0], window[1])

        # parts of train set and val set overlap by 1 row, should not be a major leak problem,
        # but if it is, it needs to be fixed
        df_t = df_t.append(df.iloc[df_v.index[-1]:ixs[0] + 1])
        df_v = df_v.append(df.iloc[ixs[0]:ixs[1] + 1])

        print(datetime.fromtimestamp(df['openTimestamp'][ixs[0]] / 1000), "-",
              datetime.fromtimestamp(df['openTimestamp'][ixs[1]] / 1000))
        print(f"Val window of {ixs[1] - ixs[0]} candles")

    df_t = df_t.append(df.iloc[df_v.index[-1]:df.index[-1]])
    df_v = df_v.drop([0])

    print(f"\nTrain set candles: {len(df_t.index)}")
    print(f"Validation set candles: {len(df_v.index)}")
    print(f"Val: {round(100 * len(df_v.index) / len(df), 2)} %")

    # droppin and cleaning
    df_t.drop(columns=['openTimestamp'], inplace=True)
    df_v.drop(columns=['openTimestamp'], inplace=True)

    print("...separating done\n")
    return df_t, df_v


def index_window(end, target_time, delta_time):  #
    target_ix = round(len(df.index) - (end / 1000 - datetime.timestamp(target_time)) / iv_sec[
        INTERVAL])  # imprecise because of missing candles

    i = target_ix
    j = target_ix
    while df['openTimestamp'][i] > 1000 * datetime.timestamp(target_time - delta_time):  # lower bound
        i -= 1
    while df['openTimestamp'][j] < 1000 * datetime.timestamp(target_time + delta_time):  # upper bound
        j += 1
    return int(i), int(j)


def scale(df_t, df_v):
    print("Scaling data...")
    # scaling
    col_names = []
    for col in df_t.columns:
        if col != "target":
            col_names.append(col)

    features = df_t[col_names]
    scaler = preprocessing.StandardScaler().fit(features.values)
    df_t[col_names] = scaler.transform(features.values)
    del features

    features = df_v[col_names]
    # scaler = preprocessing.StandardScaler().fit(features.values)
    df_v[col_names] = scaler.transform(features.values)
    del features

    print("Scaler means: ", scaler.mean_)
    print("Scaler scales: ", scaler.scale_)

    try:
        os.makedirs(f'SCALERS/{INTERVAL}-{SPECIAL_NAME}-{date_str}')

    except  FileExistsError:
        pass

    pickle_out = open(f"SCALERS/{INTERVAL}-{SPECIAL_NAME}-{date_str}/{SAVE_NAME}-scaler_data.pickle", "wb")
    pickle.dump([scaler.mean_, scaler.scale_], pickle_out)
    pickle_out.close()
    print(f'Saved the scaler')

    print("...scaling done\n")
    return df_t, df_v


def create_seqs(df, which, equalize=True):
    print(f"Creating {which} sequences...")

    sliding_window = deque(maxlen=PAST_SEQ_LEN)

    # it is easier to split between two to balance out later
    buys = []
    sells = []

    prev_index = df.index[0] - 1
    values = df.values

    pb = Progressbar(len(df.index), name="Sequentialisation")

    for i in range(len(df.index)):

        if prev_index + 1 != df.index[i]:
            sliding_window.clear()  # if not consecutive, the sliding window is bad
        prev_index = df.index[i]

        # adds single row of everything except target to sliding window, target is added at last
        sliding_window.append([n for n in values[i][:-1]])

        if len(sliding_window) == PAST_SEQ_LEN:  # when sliding window is of desired lengt
            if values[i][-1] == 0:  # sells
                sells.append([np.array(sliding_window), values[i][-1]])
            elif values[i][-1] == 1:  # buys
                buys.append([np.array(sliding_window), values[i][-1]])
                # if target == None then nothing happpens

        pb.update(i)

    del pb

    # now we can and have to shuffle
    random.shuffle(buys)
    random.shuffle(sells)

    pct_buys = round(100 * len(buys) / (len(buys) + len(sells)))
    print(f'Created {which} seqs, ({len(buys)} buys, {len(sells)} sells - {pct_buys}%/{100 - pct_buys}%)')

    log.log(f"{pct_buys}%/{100 - pct_buys}%")
    # we need to balance out
    lower = min(len(buys), len(sells))

    if equalize:
        buys = buys[:lower]
        sells = sells[:lower]

        print(f'Equalized both {which} buys/sells to {lower}')
    else:
        print('Not equalized')

    sequential_data = buys + sells
    del buys
    del sells

    random.shuffle(sequential_data)  # shuffle again

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    print(f'Returning {which} sequences ({len(y)})')
    log.log(len(y))

    print(f"...creating {which} sequences done\n")
    return np.array(X), y


def main():
    df = load_raw_data()

    df = classify(df)

    df = to_pct(df)

    df_t, df_v = separate(df)
    del df

    df_t, df_v = scale(df_t, df_v)

    train_x, train_y = create_seqs(df_t, "train")
    del df_t

    val_x, val_y = create_seqs(df_v, 'val')
    del df_v

    # print('--------------------------------------')
    print(f"Train data len: {len(train_y)}, Validation data len: {len(val_y)}")
    print(f"TRAIN:   Sells: {train_y.count(0)}, Buys: {train_y.count(1)}")
    print(f"TEST:    Sells: {val_y.count(0)}, Buys: {val_y.count(1)}")
    print(f"Validation is {round(100 * len(val_y) / (len(val_y) + len(train_y)), 2)}% of all data")
    # print('Sample: \n',  train_y[5], train_x[5])

    log.log(f"{round(100 * len(val_y) / (len(val_y) + len(train_y)), 2)}%")

    print('\nSaving training data...')

    try:
        os.makedirs(f'TRAIN_DATA/{INTERVAL}-{SPECIAL_NAME}-{date_str}')

    except  FileExistsError:
        pass

    pickle_out = open(f"TRAIN_DATA/{INTERVAL}-{SPECIAL_NAME}-{date_str}/{SAVE_NAME}-t.pickle", "wb")
    pickle.dump((train_x, train_y), pickle_out)
    pickle_out.close()

    pickle_out = open(f"TRAIN_DATA/{INTERVAL}-{SPECIAL_NAME}-{date_str}/{SAVE_NAME}-v.pickle", "wb")
    pickle.dump((val_x, val_y), pickle_out)
    pickle_out.close()

    print('...training data saved as: ')
    print(f'TRAIN_DATA/{INTERVAL}-{SPECIAL_NAME}-{date_str}/{SAVE_NAME}')

#MAIN kinda
date_str = datetime.now().strftime("%d.%m.%y")
for past in pasts:
    for future in futures:
        for pct in pcts:
            global PAST_SEQ_LEN, FUTURE_CHECK_LEN, TARGET_CHANGE
            PAST_SEQ_LEN = past
            FUTURE_CHECK_LEN = future
            TARGET_CHANGE = pct / 100

            SAVE_NAME = f'{SYMBOL}{INTERVAL}-{PAST_SEQ_LEN}x{FUTURE_CHECK_LEN}~{TARGET_CHANGE}-{SPECIAL_NAME}'
            print(f"\n\n ----- Preprocessing {SAVE_NAME} ----- ")
            log = Logger([SYMBOL, INTERVAL, SPECIAL_NAME, PAST_SEQ_LEN, FUTURE_CHECK_LEN, TARGET_CHANGE])
            main()

            log.save(f'PREP_LOG', f'{INTERVAL}-{SPECIAL_NAME}-{date_str}.csv')
            del log