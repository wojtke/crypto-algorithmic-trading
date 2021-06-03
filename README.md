# ML-risk-analysis
### It is my first project of more extensive scope. This is where I acknowledged my interest in both Machine learning and algorythmic trading. This project consists of a rather simple LSTM recurrent neural network builder (using Keras). The goal of the neural network is to predict cryptocurenncy price movements - preferrably short term. And it succeeded to do so.
#### How do I know how well the model will perform when utilized for trading in the future? I don't. I can only look on historical data and try to determine whether it has potential to be effective and make money regardless of market behavior. A big part of my work on this project was to construct tools that best describe model's performance and robustness. Using plotly I made charts that visualise model's prediction pattern and it's accuracy. Also, I made a backtesting simulation of a simple strategies that can be based on model's prediction - and visualisation of how profitable they would be over time. Lastly, I implemented a simple trading bot that can preprocess data, run predictions and send orders to the exchange (Binance) via API.

### Project structure:
 - #### Data processing and model training:
     - **getdata.py - tool used to download historical data from the exchange**
     - **new_preprocess.py - front to data preprocessing**
     - **compile.py - model training**
     - vars.py - some constant variables
     - utils.py - utilities
     - data_processing.py - data processing functions - used by new_preprocess.py, dist_acc_graph.py, predictions_graph.py
     
 - #### Visualisation:
     - **dist_acc_graph.py - model's distribution of accuracy chart**
     - **predictions_graph.py - model's chart of predictions plotted along price chart**
 
 - #### Backtesting:
     - **backtesting/brain.py - place to set a strategy and run a backtest**
     - backtesting/heart.py - 'inner' processies of backtest
     - backtesting/statistics.py - analisys and saving trades
     - backtesting/chart.py - visualisation of strategy's performance over time
     - backtesting/chart_all.py - visualisation of multiple strategies' performance over time
 
 - #### Realtime czyli w zasadzie bot tradingowy: 
     - **realtime/BRAIN.py -  place to set a strategy and run a bot**
     - realtime/HEART.py - account endpoint, placing and monitoring orders
     - realtime/GENERAL.py - ulitities

<br/>

## Charts

### dist_acc_graph.py
#### Bit more thorought analysis of model's accuracy based on how certain the model is of its prediction
![dist and acc](https://user-images.githubusercontent.com/53000695/91171365-ab55b000-e6da-11ea-85ed-d756ab82ad5d.PNG)

<br/>

### predictions_graph.py
#### Visualisation of model's predictions along with the price chart at the time.
![pred](https://user-images.githubusercontent.com/53000695/91171377-ae50a080-e6da-11ea-804b-5e78692bb1a8.PNG)

<br/>

### backtesting/chart.py
#### Profitability chart of a strategy with paremeters: 
```
    LEVERAGE = 7
    ORDER_SIZE = 0.1
    PYRAMID_MAX = 1
    THRESHOLD = 0.08
```
![trades](https://user-images.githubusercontent.com/53000695/91171387-b1e42780-e6da-11ea-81ae-f184065234c2.PNG)

<br/>

### backtesting/chart_all.py
#### Comparison on strategies with paremeters:
```
    thresholds = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16]
    leverages = [7]
    order_sizes = [0.1]
    pyramid_maxes = [1]
```
![strats](https://user-images.githubusercontent.com/53000695/91171382-b01a6400-e6da-11ea-8df8-2c8af0165bf2.PNG)
