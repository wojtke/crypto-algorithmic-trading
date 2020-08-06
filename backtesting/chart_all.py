import os
import pandas as pd
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import numpy as np

folder_docelowy = "STATISTICS"
search = "BTC15m"

SYMBOL = 'BTC'
INTERVAL = '15m'
klines_path = f'D:/PROJEKTY/Python/BINANCE_RAW_DATA/Binance_{SYMBOL}USDT_{INTERVAL}.json'

WINDOWS = [ [parse_datetime("2020"+"02"+"23"), timedelta(days=15)],
            [parse_datetime("2020"+"04"+"10"), timedelta(days=15)], ] 

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

def divide(side):
    ind = []
    if side == "SHORT":
        for i, s in enumerate(trades["Side"]):
            if s == "LONG":
                ind.append(i)
    elif side == "LONG":
        for i, s in enumerate(trades["Side"]):
            if s == "SHORT":
                ind.append(i)

    return trades.drop(ind)

def chart_candlestick():
    fig.add_trace(go.Candlestick(x=pd.to_datetime(klines[0], unit='ms'),
                    open=klines[1],
                    high=klines[2],
                    low=klines[3],
                    close=klines[4],
                    name = SYMBOL), row=1, col=1)

    fig.update_xaxes(rangeslider_visible=False)

def chart_lines():
    fig.add_shape(
            type="line",
            x0=klines[0].iloc[0],
            y0=0,
            x1=klines[0].iloc[-1],
            y1=0,
            line=dict(
                color="skyBlue",
                width=2,
                dash="dashdot"
                ),
            row=2, col=1,)
    '''
    fig.add_shape(
            type="line",
            x0=klines[0].iloc[0],
            y0=-100,
            x1=klines[0].iloc[-1],
            y1=-100,
            line=dict(
                color="salmon",
                width=2,
                dash="dashdot"
                ),
            row=2, col=1,)   
    '''

def chart_avg_balance_change_over_time(df, period, res, name): #period, accuracy w dniach liczyłbym
    real_pnl = []
    for i in df.index:
        a=1
        for e in ast.literal_eval(df['Amount'][i]):
            a*=(1-e)
        real_pnl.append((1-a)*df['PNLrealised'][i])


    final_arr = []
    count_arr = []
    time_arr = []
    ts=df['Exit time'][0]+period*24*60*60*1000
    i=0
    going = True
    while going:
        a=1
        while df['Exit time'][i]>ts-period*24*60*60*1000:
            i-=1
        i+=1
        while df['Exit time'][i]<ts:
            a*=(real_pnl[i]+1)
            i+=1
            if i>=len(df.index):
                going = False
                break
        final_arr.append(100*(a-1))
        time_arr.append(ts)
        ts+=res*24*60*60*1000

    fig.add_trace(go.Scatter(
                hovertemplate = f'{name}'+
                                '<br>Change: %{y:.2f}'+
                                '<br>Date: %{x}<br>',
                x=pd.to_datetime(time_arr, unit="ms"),
                y=final_arr),
                row=2, col=1)

klines = get_data()

fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(SYMBOL+" price "+INTERVAL, "Trades "+folder_docelowy+" "+search),
            shared_xaxes=True,
            vertical_spacing =0.05,
            row_heights=[0.3, 0.7])


chart_lines()
chart_candlestick()
for d in os.listdir(folder_docelowy):
    if d.find(search)>=0:
        print(folder_docelowy+"\\"+d+"\\trades.csv")
        df = pd.read_csv(folder_docelowy+"\\"+d+"\\trades.csv")
        try:
            chart_avg_balance_change_over_time(df, 7, 1, d)
        except:
            print("error so im loading next")


fig.show()



