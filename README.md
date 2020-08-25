# ML-risk-analysis
### To jest narzędzie do tworzenia prostych sieci neuronowych LSTM dla analizy ryzyka / przewidywania ruchów na giełdzie kryptowalut (Binance) oraz narzędzi diagnostycznych, które potrafią sprawność tych sieci przeanalizować. Jest to bardziej data science project niż machine learning project. Sieć neuronowa jest tutaj tylko wisienką na torcie.

<br/>

### Co jest odpowiedzialne za co:
 - #### Przygotowanie danych i trenowanie sieci:
     - **getdata.py - pobiera/zapisuje/aktualizuje historyczne dane z giełdy**
     - **new_preprocess.py - przygotowuje dane treningowe**
     - **compile.py - trenuje model**
     - vars.py - stałe zmienne
     - utils.py - utilities
     - data_processing.py - tam cała struktura przetwarzania danych jest, tego modułu używa new_preprocess.py oraz wykresy dist_acc_graph.py, predictions_graph.py
     
 - #### Wizualizacja sprawności modelu:
     - **dist_acc_graph.py - wykres trafności predykcji danego modelu sieci neuronowej**
     - **predictions_graph.py - wykres predykcji danego modelu sieci neuronowej**
 
 - #### Backtesting:
     - **backtesting/brain.py - ustalamy tu strategię i odpalamy backtest**
     - backtesting/heart.py - tu wykonują się 'wewnętrzne' procesy backtestingu, kiedy w brain.py jedynie podejmowane są decyzje
     - backtesting/statistics.py - moduł odpowiedzialny za analizę i zapis zagrań dla danej strategii
     - backtesting/chart.py - plik, kopiowany do każdego z folderów w STATISTICS/ za pomocą backtesting/update_chart_files.py, pozwalający zwizualizować sprawność danej strategii w czasie
     - backtesting/chart_all.py - wykres, na którym zwizualizowana jest sprawność wielu strategii jednocześnie, pozwalając na ich porównanie
 
 - #### Realtime czyli w zasadzie bot tradingowy: 
     - **realtime/BRAIN.py - ustalamy tu strategię i odpalamy bota tradingowego pracującego na prawdziwej giełdzie już**
     - realtime/HEART.py - account endpoint, wystawiający i monitorujący zlecenia
     - realtime/GENERAL.py - takie ulitities
 
 
### Wykresy

#### dist_acc_graph.py
#### Dokładna statystyczna analiza trafności predykcji sieci, pozwala ocenić znacznie więcej niż tylko val_loss i val_accuracy (na danych testowych)
![dist and acc](https://user-images.githubusercontent.com/53000695/91171365-ab55b000-e6da-11ea-85ed-d756ab82ad5d.PNG)

<br/>

#### predictions_graph.py
#### Wizualizacja, jak sieć przewiduje zmianę ceny w danej sytuacji (na danych testowych oczywiście)
![pred](https://user-images.githubusercontent.com/53000695/91171377-ae50a080-e6da-11ea-804b-5e78692bb1a8.PNG)

<br/>

#### backtesting/chart.py
#### Konkretny wykres dla strategi o parametrach: 
```
    LEVERAGE = 7
    ORDER_SIZE = 0.1
    PYRAMID_MAX = 1
    THRESHOLD = 0.08
```
![trades](https://user-images.githubusercontent.com/53000695/91171387-b1e42780-e6da-11ea-81ae-f184065234c2.PNG)

<br/>

#### backtesting/chart_all.py
#### Porównanie wykresów dla strategi o parametrach:
```
    thresholds = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16]
    leverages = [7]
    order_sizes = [0.1]
    pyramid_maxes = [1]
```
![strats](https://user-images.githubusercontent.com/53000695/91171382-b01a6400-e6da-11ea-8df8-2c8af0165bf2.PNG)
