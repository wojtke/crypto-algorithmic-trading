from tensorflow.keras.models import load_model
import pickle 
from numpy import argmax
import plotly.graph_objects as go
from plotly.subplots import make_subplots




MODEL = '15m-normal-04.07.20/BTCUSDT15m-100x30~0.005-normal-04Jul20-15.27.18/17-TL0.536-TA0.640_VL0.596-VA0.591.model'

MODEL_PATH = "D:/PROJEKTY/Python/ML risk analysis/MODELS/" + MODEL
TEST_PATH = "D:/PROJEKTY/Python/ML risk analysis/TRAIN_DATA/" + MODEL[:-58] +'-v.pickle'

div = 50



def czy_poprawne(entry, i):
    if argmax(entry) == validation_y[i]:
        return True
    else:
        return False


model = load_model(MODEL_PATH, compile=False)

with open(TEST_PATH , 'rb') as pickle_in:
    validation_x, validation_y = pickle.load(pickle_in)

prediction = model.predict([validation_x])
print(prediction)



right=[]
for i, entry in enumerate(prediction):
    right.append(czy_poprawne(entry, i))

#inicjalizacja array
distribution = [0] * div
accuracy = [0] * div
accuracy_threshhold = [0] * div
distribution_threshhold = [0] * div
aggregate_accuracy_threshhold = [0] * int(div/2)
aggregate_distribution_threshhold = [0] * int(div/2)

#proste rozłożenie
for i in range(len(right)):
    place = int(prediction[i][0]*div)
    distribution[place]+=1
    if right[i]:
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
    accuracy[i]=100*accuracy[i]/distribution[i]
    distribution[i] = 100*distribution[i]/m
    accuracy_threshhold[i]=100*accuracy_threshhold[i]/distribution_threshhold[i]

#zmiana accucarcy z liczby poprawnych, na procent poprawnych w przypadku aggregate oraz zmiana wszelkich distribution threshhold na procentowe
for i in range(int(div/2)):
    aggregate_accuracy_threshhold[i] = 100*aggregate_accuracy_threshhold[i]/aggregate_distribution_threshhold[i]

    distribution_threshhold[i]=100*distribution_threshhold[i]/distribution_threshhold[int(div/2)-1]
    distribution_threshhold[div-i-1]=100*distribution_threshhold[div-i-1]/distribution_threshhold[int(div/2)]
    aggregate_distribution_threshhold[i]=100*aggregate_distribution_threshhold[i]/aggregate_distribution_threshhold[int(div/2)-1]    



fig = make_subplots(rows=1, cols=2, 
                    column_widths=[0.5, 0.5], 
                    horizontal_spacing=0.07,
                    specs=[[{}, {}]])


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
    title=MODEL,
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

