import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import os
import pickle
import numpy as np

from utils import Progressbar
from data_processing import Preprocessor


                     
MODEL = '15m-ehh-24.08.20/BTCUSDT15m-100x50~0.01-24Aug20-23.26.47/05-TL0.690-TA0.533_VL0.687-VA0.533.model'

FOLDER = '15m-new-10.08.20/BTCUSDT15m-100x50~0.01-11Aug20-12.33.21/'
if FOLDER[-1]!="/":
    FOLDER+="/"



def main():
    preprocessor = Preprocessor()
    preprocessor.klines_load()

    graph(MODEL, preprocessor, name=MODEL)
    #many_graphs(FOLDER, preprocessor, thing="VA", value=0.56, mode="max")


def many_graphs(FOLDER, preprocessor, thing="VL", value=0.7, mode="min"):
    DIR = Vars.main_path + "MODELS/" + FOLDER
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
        MODEL = FOLDER + model_name
        print(MODEL)

        graph(MODEL, preprocessor, name=MODEL)    


def chart_candlestick(fig, preprocessor):
    fig.add_trace(go.Candlestick(x=pd.to_datetime(preprocessor.klines['openTimestamp'], unit='ms'),
                    open=preprocessor.klines['open'],
                    high=preprocessor.klines['high'],
                    low=preprocessor.klines['low'],
                    close=preprocessor.klines['close'],
                    name = preprocessor.SYMBOL), row=1, col=1)

    fig.update_xaxes(rangeslider_visible=False)


def chart_lines(fig, start, end):
    #linia 50% pewności
    fig.add_shape(
            type="line",
            x0=start,
            y0=0.5,
            x1=end,
            y1=0.5,
            line=dict(
                color="skyBlue",
                width=2,
                dash="dashdot"
                ),
            row=2, col=1,)

    #linie wyznaczające gdzie jest validation perion 
    '''
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
    '''


def chart_model_pred(fig, MODEL, preprocessor):
    preprocessor.repreprocess(MODEL)

    pred_right, ts_right, pred_wrong, ts_wrong, pred_none, ts_none = preprocessor.analyze_predictions()

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


def graph(MODEL, preprocessor, name):
    fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(preprocessor.SYMBOL+" price "+ preprocessor.INTERVAL, "Pred "+ os.path.basename(os.getcwd())),
                shared_xaxes=True,
                vertical_spacing =0.06,
                row_heights=[0.45, 0.55])

    chart_candlestick(fig, preprocessor)
    chart_lines(fig, preprocessor.klines['openTimestamp'].iloc[0], preprocessor.klines['openTimestamp'].iloc[-1])

    chart_model_pred(fig, MODEL, preprocessor)

    fig.update_layout(legend_orientation="h", title=name)
    fig.show()


main()