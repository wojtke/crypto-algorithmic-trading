import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import ast
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import os
from tensorflow import keras
import pickle
from collections import deque
from sklearn import preprocessing
from progressbar import Progressbar
import numpy as np


                     
MODEL = '15m-normal-04.07.20/BTCUSDT15m-100x30~0.005-normal-04Jul20-15.27.18/17-TL0.536-TA0.640_VL0.596-VA0.591.model'

MODEL_PATH = "D:/PROJEKTY/Python/ML risk analysis/MODELS/" + MODEL
SCALER_PATH = "D:/PROJEKTY/Python/ML risk analysis/SCALERS/" + MODEL[:-58] + '-scaler_data.pickle'

WINDOWS = [ [parse_datetime("2020"+"02"+"23"), timedelta(days=15)],
                [parse_datetime("2020"+"04"+"10"), timedelta(days=15)], ] 

SYMBOL = 'BTC'
INTERVAL = '15m'
PAST_SEQ_LEN = 100
klines_path = os.path.normpath(os.getcwd()) + f'\\RAW_DATA\\Binance_{SYMBOL}USDT_{INTERVAL}.json'
print(klines_path)


marker_color = {'SHORT': "tomato", "LONG": "lime", "BOTH": 'yellow'}
marker_color_dark = {'SHORT': "crimson", "LONG": "forestGreen", "BOTH": 'gold'}
marker_symbol = {'SHORT': "triangle-down", "LONG": "triangle-up"}


def chart_candlestick():
    fig.add_trace(go.Candlestick(x=pd.to_datetime(klines[0], unit='ms'),
                    open=klines[1],
                    high=klines[2],
                    low=klines[3],
                    close=klines[4],
                    name = SYMBOL), row=1, col=1)

    fig.update_xaxes(rangeslider_visible=False)


def chart_lines():
    #linia 50% pewności
    fig.add_shape(
            type="line",
            x0=klines[0].iloc[0],
            y0=0.5,
            x1=klines[0].iloc[-1],
            y1=0.5,
            line=dict(
                color="skyBlue",
                width=2,
                dash="dashdot"
                ),
            row=2, col=1,)

    #linie wyznaczające gdzie jest validation perion 

    for w in WINDOWS:
        ts1 = 1000*datetime.timestamp(w[0] - w[1])
        ts2 = 1000*datetime.timestamp(w[0] + w[1])
        fig.add_shape(
                type="line",
                x0=ts1,
                y0=4000,
                x1=ts1,
                y1=12000,
                line=dict(
                    color="blue",
                    width=1,
                    dash="dashdot"
                    ),
                row=1, col=1,)

        fig.add_shape(
                type="line",
                x0=ts2,
                y0=4000,
                x1=ts2,
                y1=12000,
                line=dict(
                    color="blue",
                    width=1,
                    dash="dashdot"
                    ),
                row=1, col=1,)        


def get_data():
    klines_uncut = pd.read_json(klines_path)
    klines = pd.DataFrame()

    for w in WINDOWS: 
        ts1 = 1000*datetime.timestamp(w[0] - w[1])
        ts2 = 1000*datetime.timestamp(w[0] + w[1])
        a = np.array(klines_uncut.loc[klines_uncut[0].isin([ts1, ts2])].index)

        if klines.empty:
            klines = klines_uncut.iloc[a[0]:a[1]]
        else:
            klines = klines.append(klines_uncut.iloc[a[0]:a[1]])
    return klines


def process_data(df):
    df = df[[0,1,2,3,4,5]]
    df.columns = ['openTimestamp', 'open', 'high', 'low', 'close', 'volume']

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    #df = df.reset_index()
    #df = df.drop(columns=["index"])

    print("...loading done\n")
    ########################################################
    print("Changing values to pct change...")   

    pre_dropna= len(df.index)
    df.dropna(subset=['close', 'volume', 'high', 'low'], inplace=True)

    df['low'] = df['close']/df['low'] #low wick
    df['high'] = df['high']/df['close'] #high wick

    df['close'] = df['close'].pct_change()
    df['volume'] = df['volume'].add(1) #to jest po to, zeby nie wywalac swiec o volume 0, bo wtedy pct_change -> inf
    df['volume'] = df['volume'].pct_change() 

    df.drop(columns=['open'], inplace=True)  #drop unused columns except openTimestamp

    df.dropna(subset=['close', 'volume', 'high', 'low'], inplace=True)
    print(f"Dropped {pre_dropna - len(df.index)} rows")

    print("...changing to pct done\n")
    ########################################################
    pickle_scaler_data = pickle.load( open( SCALER_PATH, "rb" ) )
    print(pickle_scaler_data)

    i=0
    for col in df.columns: 
        if col != "openTimestamp":
            df[col] = df[col].sub(pickle_scaler_data[0][i]).div(pickle_scaler_data[1][i])
            i+=1

    ########################################################
    sliding_window = deque(maxlen=PAST_SEQ_LEN) 

    #it is easier to split between two to balance out later
    seqs=[]
    ts =[]

    prev_index = df.index[0]-1

    pb = Progressbar(len(df.index), name="Sequentialisation")

    for i in range(len(df.index)-1): 

        sliding_window.append([n for n in df.values[i][1:]]) #adds single row of everything except target to sliding window, target is added at last

        if len(sliding_window) == PAST_SEQ_LEN: #when sliding window is of desired length
            seqs.append(np.array(sliding_window))
            ts.append(df.values[i][0])

        if df.index[i]+1!=df.index[i+1]:
            sliding_window.clear()

        pb.update(i)
    del pb



    return np.array(seqs), ts


def chart_model_pred():
    seqs, ts = process_data(klines)    

    model = keras.models.load_model(MODEL_PATH)

    print([seqs])
    prediction = model.predict([seqs])

    pred = []
    for row in prediction:
        pred.append(row[0])

    fig.add_trace(go.Scatter(
                name = "pred",
                mode='markers',
                x=pd.to_datetime(ts, unit="ms"), 
                y=pred),
                row=2, col=1)
    '''

    df = divide(side)
    if len(df.index)<n:
        return 0

    real_pnl = []
    for i in df.index:
        a=1
        for e in ast.literal_eval(df['Amount'][i]):
            a*=(1-e)
        real_pnl.append((1-a)*df['PNLrealised'][i])


    average_pnl = []

    for i in range(len(real_pnl)-n+1):
        a=1
        for e in real_pnl[i:i+n]:
            a*=(e+1)
        average_pnl.append(100*(a-1))


    '''




fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(SYMBOL+" price "+ INTERVAL, "Pred "+ os.path.basename(os.getcwd())),
            shared_xaxes=True,
            vertical_spacing =0.06,
            row_heights=[0.45, 0.55])




klines = get_data()

chart_candlestick()
chart_lines()

chart_model_pred()



fig.update_layout(legend_orientation="h", title=MODEL)
fig.show()

