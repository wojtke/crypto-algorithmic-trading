# ML-risk-analysis
### To jest narzędzie do tworzenia prostych sieci neuronowych LSTM dla analizy ryzyka / przewidywania ruchów na giełdzie kryptowalut (Binance) oraz narzędzi diagnostycznych, które potrafią sprawność tych sieci przeanalizować.

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
 
 
