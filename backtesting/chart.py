import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import ast
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import os
import numpy as np

SYMBOL = 'BTC'
INTERVAL = '1h'

klines_path = klines_path = f'D:/PROJEKTY/Python/BINANCE_RAW_DATA/Binance_{SYMBOL}USDT_{INTERVAL}.json'
#path = f'D:\\PROJEKTY\\Python\\RSI Boll Binance Bot\\RAW_DATA\\Binance_{SYMBOL}USDT_{INTERVAL}.json'

WINDOWS = [ [parse_datetime("2020"+"02"+"23"), timedelta(days=15)],
            [parse_datetime("2020"+"04"+"10"), timedelta(days=15)], ] 

marker_color = {'SHORT': "tomato", "LONG": "lime", "BOTH": 'yellow'}
marker_color_dark = {'SHORT': "crimson", "LONG": "forestGreen", "BOTH": 'gold', "BOTHBIG": 'orange'}
marker_symbol = {'SHORT': "triangle-down", "LONG": "triangle-up"}

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

    to_return=trades.drop(ind)
    to_return = to_return.reset_index()
    to_return = to_return.drop(columns=["index"])

    return to_return

def chart_candlestick():
    fig.add_trace(go.Candlestick(x=pd.to_datetime(klines[0], unit='ms'),
                    open=klines[1],
                    high=klines[2],
                    low=klines[3],
                    close=klines[4],
                    name = SYMBOL), row=1, col=1)

    fig.update_xaxes(rangeslider_visible=False)

def chart_lines():
    #zero line
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
    #liq line
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
def chart_add_exits(side):
    df = divide(side)
    fig.add_trace(go.Scatter(x=pd.to_datetime(df['Exit time'], unit='ms'), y=df['Exit price'],
                        mode='markers',
                        marker_symbol='circle',
                        marker_color=marker_color[side],
                        marker_line_width=1,
                        marker_line_color='black',
                        marker_size=4,
                        name=side))



def chart_add_entries(side):
    time = []
    price = []
    for i in trades.index:
        if trades["Side"][i]==side:
            for a in ast.literal_eval(trades["Entry price"][i]):
                price.append(a)

            for a in ast.literal_eval(trades["Entry time"][i]):
                time.append(a)


    fig.add_trace(go.Scatter(x=pd.to_datetime(time, unit='ms'), y=price,
                        mode='markers',
                        marker_symbol=marker_symbol[side],
                        marker_color=marker_color[side],
                        marker_line_width=1,
                        marker_line_color='black',
                        marker_size=8,
                        name=side))

def chart_pnl_per_trade(side):
    df = divide(side)

    fig.add_trace(go.Scatter(
                name = f"PNL % per {side}",
                x=pd.to_datetime(df["Exit time"], unit="ms"), 
                y=100*df["PNLrealised"],
                mode='markers',
                marker_color=marker_color[side],
                error_y=dict(
                type='data',
                symmetric=False,
                array=100*(df["PNLmax"] - df["PNLrealised"]),
                arrayminus=100*(df["PNLrealised"] - df["PNLmin"]))),
                row=2, col=1,)

def chart_avg_balance_change_over_time(side, period, res, width): #period, accuracy w dniach liczyłbym
    df=divide(side)

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
        count=0
        while df['Exit time'][i]>ts-period*24*60*60*1000:
            i-=1
        i+=1
        while df['Exit time'][i]<ts:
            a*=(real_pnl[i]+1)
            i+=1
            count+=1
            if i>=len(df.index):
                going = False
                break
        count_arr.append(count)
        final_arr.append(100*(a-1))
        time_arr.append(ts)
        ts+=res*24*60*60*1000

    fig.add_trace(go.Scatter(
                name = f"Balance %\nchange over\nlast {period} days (side: {side})",
                line=dict(
                    color=marker_color_dark[side],
                    width=width,
                ),
                x=pd.to_datetime(time_arr, unit="ms"),
                y=final_arr),
                row=2, col=1)

def chart_avg_balance_change(side, n):

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

    fig.add_trace(go.Scatter(
                name = f"Balance %\nchange over\nlast {n} trades (side: {side})",
                marker_color=marker_color_dark[side],
                x=pd.to_datetime(df["Exit time"][n-1:], unit="ms"), 
                y=average_pnl),
                row=2, col=1)


fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(SYMBOL+" price "+INTERVAL, "Trades "+ os.path.basename(os.getcwd())),
            shared_xaxes=True,
            vertical_spacing =0.06,
            row_heights=[0.45, 0.55])


trades = pd.read_csv("trades.csv")
print(trades)

klines = get_data()

chart_candlestick()
chart_lines()

'''
chart_pnl_per_trade("LONG")
chart_pnl_per_trade("SHORT")
'''
chart_add_exits("SHORT")
chart_add_exits("LONG")

chart_add_entries("SHORT")
chart_add_entries("LONG")

'''
chart_avg_balance_change("SHORT", int(len(trades.index)/12))
chart_avg_balance_change("LONG", int(len(trades.index)/12))
chart_avg_balance_change("BOTH", int(len(trades.index)/9))
'''

chart_avg_balance_change_over_time("SHORT", 3, 1, width=1)
chart_avg_balance_change_over_time("LONG", 3, 1, width=1)
chart_avg_balance_change_over_time("BOTH", 3, 1, width=2)
chart_avg_balance_change_over_time("BOTHBIG", 9, 1, width=2)


fig.update_layout(legend_orientation="h")
fig.show()

