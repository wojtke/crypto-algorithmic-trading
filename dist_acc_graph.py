import pickle 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from numpy import argmax

from data_processing import Preprocessor



MODEL = '15m-ehh-24.08.20/BTCUSDT15m-100x50~0.01-24Aug20-23.26.47/05-TL0.690-TA0.533_VL0.687-VA0.533.model'

FOLDER = '15m-new-10.08.20/BTCUSDT15m-100x50~0.01-11Aug20-12.33.21/'
if FOLDER[-1]!="/":
    FOLDER+="/"

div = 50



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



def graph(MODEL, preprocessor, name):
    preprocessor.repreprocess(MODEL)

    pred_bool=[]
    for pred, ts, target in preprocessor.pred_df.values:
        if target == round(pred):
                pred_bool.append(True)
        elif target + round(pred)==1:
            pred_bool.append(False)
        else:
            pred_bool.append(None)

    #inicjalizacja array
    distribution = [0] * div
    accuracy = [0] * div
    accuracy_threshhold = [0] * div
    distribution_threshhold = [0] * div
    aggregate_accuracy_threshhold = [0] * int(div/2)
    aggregate_distribution_threshhold = [0] * int(div/2)

    #proste rozłożenie
    for i in range(len(pred_bool)):
        place = int(preprocessor.pred_df['preds'].values[i]*div)
        distribution[place]+=1
        if pred_bool[i]:
            accuracy[place]+=1

    #rozłożenie progowe lewej strony
    accuracy_threshhold[0]=accuracy[0]
    distribution_threshhold[0]=distribution[0]
    for i in range(int(div/2)-1):
        accuracy_threshhold[i+1]=accuracy_threshhold[i]+accuracy[i+1]
        distribution_threshhold[i+1]=distribution_threshhold[i]+distribution[i+1]

    #rozłożenie progowe prawej strony
    accuracy_threshhold[-1]=accuracy[-1]
    distribution_threshhold[-1]=distribution[-1]
    for i in range(int(div/2)-1):
        accuracy_threshhold[div-i-2]=accuracy_threshhold[div-i-1]+accuracy[div-i-2]
        distribution_threshhold[div-i-2]=distribution_threshhold[div-i-1]+distribution[div-i-2]

    #złożenie rozłożonych prawej i lewej strony
    for i in range(int(div/2)):
        aggregate_distribution_threshhold[i]+=distribution_threshhold[i]+distribution_threshhold[div-i-1]
        aggregate_accuracy_threshhold[i]+=accuracy_threshhold[i]+accuracy_threshhold[div-i-1]

    #zmiana accucarcy z liczby poprawnych, na procent poprawnych
    m=distribution[argmax(distribution)]
    for i in range(div):
        if accuracy[i]>0:
            accuracy[i]=100*accuracy[i]/distribution[i]
        else:
            accuracy[i] = None

        distribution[i] = 100*distribution[i]/m

        if accuracy_threshhold[i]>0:
            accuracy_threshhold[i]=100*accuracy_threshhold[i]/distribution_threshhold[i]
        else:
            accuracy_threshhold[i] = None
        

    #zmiana accucarcy z liczby poprawnych, na procent poprawnych w przypadku aggregate oraz zmiana wszelkich distribution threshhold na procentowe
    for i in range(int(div/2)):
        if aggregate_accuracy_threshhold[i]>0:
            aggregate_accuracy_threshhold[i] = 100*aggregate_accuracy_threshhold[i]/aggregate_distribution_threshhold[i]
        else:
            aggregate_accuracy_threshhold[i] = None

        distribution_threshhold[i]=100*distribution_threshhold[i]/distribution_threshhold[int(div/2)-1]
        distribution_threshhold[div-i-1]=100*distribution_threshhold[div-i-1]/distribution_threshhold[int(div/2)]

        aggregate_distribution_threshhold[i]=100*aggregate_distribution_threshhold[i]/aggregate_distribution_threshhold[int(div/2)-1]    



    fig = make_subplots(rows=1, cols=2, 
                        column_widths=[0.5, 0.5], 
                        horizontal_spacing=0.07,)


    fig.add_trace(
        go.Bar(x=[(x/div) for x in range(div)], y=distribution, name="Distribution"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=[(x/div) for x in range(div)], y=accuracy, name="Accuracy", mode='markers'),
        row=1, col=1
    )





    fig.add_trace(
        go.Scatter(x=[(x/div) for x in range(int(div/2))], 
            y=distribution_threshhold[int(div/2):], 
            name="Distribution threshhold sells",
            line = dict(color='#b7850b', width=2, dash='dashdot')),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scatter(x=[(x/div) for x in range(int(div/2))],
            y=distribution_threshhold[::-1][int(div/2):],
            name="Distribution threshhold buys",
            line = dict(color='#239b56', width=2, dash='dashdot')),
        row=1, col=2
    )


    fig.add_trace(
        go.Scatter(x=[(x/div) for x in range(int(div/2))], 
            y=accuracy_threshhold[int(div/2):], 
            name="Accuracy threshhold sells",
            line = dict(color='#ca6f1e', width=2, dash='dashdot')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=[(x/div) for x in range(int(div/2))],
        y=accuracy_threshhold[::-1][int(div/2):], 
        name="Accuracy threshhold buys",
        line = dict(color='#138d75', width=2, dash='dashdot')),
        row=1, col=2
    )



    fig.add_trace(
        go.Scatter(x=[(x/div) for x in range(int(div/2))], 
        y=aggregate_distribution_threshhold[::-1], 
        name="Distribution threshhold aggregate",
        line = dict(color='#1f618d', width=3)),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=[(x/div) for x in range(int(div/2))], 
        y=aggregate_accuracy_threshhold[::-1], 
        name="Accuracy threshhold aggregate",
        line = dict(color='#6c3483', width=3)),
        row=1, col=2
    )
                

    fig.update_layout(
        title=name,
        legend_orientation="h",
        xaxis = dict(
            tickmode = 'linear',
            dtick = 2/div,
            tick0 = 0),
        xaxis2 = dict(
            tickmode = 'linear',
            dtick = 1/div,
            tick0 = 0),
        yaxis = dict(
            tickmode = 'linear',
            dtick = 5,
            tick0 = 0),
        yaxis2 = dict(
            tickmode = 'linear',
            dtick = 5,
            tick0 = 0)
        )

    fig.show()

main()