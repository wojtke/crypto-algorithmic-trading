import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#import ast
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import os
import pickle
#from collections import deque
#from sklearn import preprocessing
from utils import Progressbar
import numpy as np

from proc_func import Preprocessor


                     
MODEL = '15m-normal-11.07.20/BTCUSDT15m-100x50~0.01-normal-11Jul20-17.59.26/07-TL0.679-TA0.564_VL0.687-VA0.563.model'

MODEL_PATH = "D:/PROJEKTY/Python/ML risk analysis/MODELS/" + MODEL
SCALER_PATH = "D:/PROJEKTY/Python/ML risk analysis/SCALERS/" + MODEL[:-58] + '-scaler_data.pickle'
READY_PRED_PATH = "D:/PROJEKTY/Python/ML risk analysis/MODELS/" + MODEL[:-5] + 'pickle'

FOLDER = '15m-normal-11.07.20/BTCUSDT15m-100x50~0.01-normal-11Jul20-17.59.26/'

if FOLDER[-1]!="/":
    FOLDER+="/"

WINDOWS = [ [parse_datetime("2020"+"02"+"23"), timedelta(days=15)],
            [parse_datetime("2020"+"04"+"10"), timedelta(days=15)], ] 

SYMBOL = 'BTC'
INTERVAL = '15m'
PAST_SEQ_LEN = 100
FUTURE_CHECK_LEN = 50
TARGET_CHANGE = 0.01


klines_path = os.path.normpath(os.getcwd()) + f'\\RAW_DATA\\Binance_{SYMBOL}USDT_{INTERVAL}.json'
print(klines_path)


def main():
    global klines
    klines = get_data()
    graph(MODEL_PATH, SCALER_PATH, name=MODEL)
    #many_graphs(FOLDER, thing="VL", value=0.69, mode="min")

def many_graphs(FOLDER, thing="VL", value=0.7, mode="min"):
    DIR = "D:/PROJEKTY/Python/ML risk analysis/MODELS/" + FOLDER
    arr=[]
    if mode=="min":
        for d in os.listdir(DIR):
            if d.find(thing) != -1:
                if float(d[d.find(thing)+2:d.find(thing)+7])<value:
                    arr.append(d)
    if mode=="max":
        for d in os.listdir(DIR):
            if d.find(thing) != -1:
                if float(d[d.find(thing)+2:d.find(thing)+7])>value:
                    arr.append(d)

    for model_name in arr:
        print(model_name)
        MODEL_PATH = "D:/PROJEKTY/Python/ML risk analysis/MODELS/" + FOLDER + model_name
        SCALER_PATH = "D:/PROJEKTY/Python/ML risk analysis/SCALERS/" + FOLDER[:-18] +'-scaler_data.pickle'
        print(MODEL_PATH)
        print(SCALER_PATH)

        graph(MODEL_PATH, SCALER_PATH, name=FOLDER + model_name)    


def chart_candlestick(fig):
    fig.add_trace(go.Candlestick(x=pd.to_datetime(klines[0], unit='ms'),
                    open=klines[1],
                    high=klines[2],
                    low=klines[3],
                    close=klines[4],
                    name = SYMBOL), row=1, col=1)

    fig.update_xaxes(rangeslider_visible=False)


def chart_lines(fig):
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


def chart_model_pred(fig, MODEL_PATH, SCALER_PATH):
    preprocessor = Preprocessor(SYMBOL, INTERVAL, MODEL, WINDOWS)
    
    preprocessor.repreprocess(do_not_use_ready=True)

    pred_right, ts_right, pred_wrong, ts_wrong, pred_none, ts_none = preprocessor.analyze_predictions()

    del preprocessor

    fig.add_trace(go.Scatter(
                name = "not classified",
                mode='markers',
                marker_color='wheat',
                x=pd.to_datetime(ts_none, unit="ms"), 
                y=pred_none),
                row=2, col=1)  

    fig.add_trace(go.Scatter(
                name = "Right pred",
                mode='markers',
                marker_color='lightGreen',
                x=pd.to_datetime(ts_right, unit="ms"), 
                y=pred_right),
                row=2, col=1)

    fig.add_trace(go.Scatter(
                name = "Wrong pred",
                mode='markers',
                marker_color='indianRed',
                x=pd.to_datetime(ts_wrong, unit="ms"), 
                y=pred_wrong),
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

def graph(MODEL_PATH, SCALER_PATH, name):
    fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(SYMBOL+" price "+ INTERVAL, "Pred "+ os.path.basename(os.getcwd())),
                shared_xaxes=True,
                vertical_spacing =0.06,
                row_heights=[0.45, 0.55])

    chart_candlestick(fig)
    chart_lines(fig)

    chart_model_pred(fig, MODEL_PATH, SCALER_PATH)

    fig.update_layout(legend_orientation="h", title=name)
    fig.show()


main()