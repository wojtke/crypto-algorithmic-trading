import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import pickle
from collections import deque
from utils import Progressbar
import os

class Preprocessor:
    def __init__(self, symbol, interval, MODEL, WINDOWS):

        self.klines_path = f'D:/PROJEKTY/Python/BINANCE_RAW_DATA/Binance_{symbol}USDT_{interval}.json'

        self.MODEL_PATH = "D:/PROJEKTY/Python/ML risk analysis/MODELS/" + MODEL
        self.SCALER_PATH = "D:/PROJEKTY/Python/ML risk analysis/SCALERS/" + MODEL[:-58] + '-scaler_data.pickle'
        self.READY_PRED_PATH = "D:/PROJEKTY/Python/ML risk analysis/READY_PRED/" + MODEL[:-5] + 'pickle'


        self.TARGET_CHANGE = 0.01
        self.FUTURE_CHECK_LEN = 50
        self.PAST_SEQ_LEN = 100

        self.klines = None


        self.WINDOWS = [ [parse_datetime("2020"+"02"+"23"), timedelta(days=15)],
                         [parse_datetime("2020"+"04"+"10"), timedelta(days=15)], ] 


    def repreprocess(self, do_not_use_ready=False):
        try:
            self.pred_df = pickle.load(open( self.READY_PRED_PATH, "rb" ))
            print("Ready saved predictions found!\n")

            if do_not_use_ready:
                print("It was specified not to use ready predictions")
                raise Exception("do_not_use_ready=True")

        except:
            print("Repreprocessing shit...")
            if not self.klines:
                self.klines = self.data_load()

            df = self.data_to_pct(self.klines.copy())

            df = self.data_scale(df)
            seqs, ts, target, pred_index = self.data_make_sequences(df)

            preds = self.data_predict(seqs)

            predx = pd.DataFrame(preds)[1]

            self.pred_df = pd.DataFrame(data={"preds":(np.array(predx)), "ts":ts, "target":target}, index=pred_index)

            self.save_ready_preds()
            print("...shit repreprocessed and saved")


    def data_load(self):
        print("Loading data...")

        klines_uncut = pd.read_json(self.klines_path)
        klines = pd.DataFrame()

        for w in self.WINDOWS: 
            ts1 = 1000*datetime.timestamp(w[0] - w[1])
            ts2 = 1000*datetime.timestamp(w[0] + w[1])
            a = np.array(klines_uncut.loc[klines_uncut[0].isin([ts1, ts2])].index)

            if klines.empty:
                klines = klines_uncut.iloc[a[0]:a[1]]
            else:
                klines = klines.append(klines_uncut.iloc[a[0]:a[1]])


        klines = klines[[0,1,2,3,4,5]]
        klines.columns = ['openTimestamp', 'open', 'high', 'low', 'close', 'volume']

        for c in klines.columns:
            klines[c] = pd.to_numeric(klines[c], errors='coerce')
        #df = df.reset_index()
        #df = df.drop(columns=["index"])

        print("...loading done\n")
        return klines


    def data_to_pct(self, df):
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
        return df


    def data_scale(self, df):
        print("Scaling data...")

        pickle_scaler_data = pickle.load( open( self.SCALER_PATH, "rb" ) )
        print(pickle_scaler_data)

        i=0
        for col in df.columns: 
            if col != "openTimestamp":
                df[col] = df[col].sub(pickle_scaler_data[0][i]).div(pickle_scaler_data[1][i])
                i+=1

        print("...scaling done\n")
        return df


    def data_make_sequences(self, df):
        print("Creating sequences...")

        sliding_window = deque(maxlen=self.PAST_SEQ_LEN) 

        seqs=[]
        ts =[]
        target=[]
        pred_index=[]

        prev_index = df.index[0]-1

        pb = Progressbar(len(df.index), name="Sequentialisation")

        for i in range(len(df.index)-1): 

            sliding_window.append([n for n in df.values[i][1:]]) #adds single row of everything except target to sliding window, target is added at last

            if len(sliding_window) == self.PAST_SEQ_LEN: #when sliding window is of desired length
                seqs.append(np.array(sliding_window))
                ts.append(df.values[i][0])
                target.append(self.assign_classification(df.index[i]))
                pred_index.append(df.index[i])

            if df.index[i]+1!=df.index[i+1]:
                sliding_window.clear()

            pb.update(i)
        del pb

        print("...creating sequences done\n")
        return np.array(seqs), ts, target, pred_index

    def data_target(self, df):
        pass

    def data_predict(self, seqs):
        print("Loading model and getting predictions...")

        from tensorflow import keras

        model = keras.models.load_model(self.MODEL_PATH)
        preds = model.predict([seqs])

        print("...done loading model and getting predictions\n")
        return preds


    def analyze_predictions(self):
        print("Analyzing predictions...")

        pred_right = []
        ts_right = []
        pred_wrong = []
        ts_wrong = []
        pred_none = []
        ts_none = []

        for pred, ts, target in self.pred_df.values:
            if target == round(pred):
                pred_right.append(pred)
                ts_right.append(ts)
            elif target + round(pred)==1:
                pred_wrong.append(pred)
                ts_wrong.append(ts)
            else:
                pred_none.append(pred)
                ts_none.append(ts)

        print("...analyzed")
        print(f"Right: {len(pred_right)}")
        print(f"Wrong: {len(pred_wrong)}")
        print(f"Null: {len(pred_none)}")

        print(f"Rights pct: {round(100*len(pred_right)/(len(pred_wrong)+len(pred_right)),2)}% " +
              f"or {round(100*len(pred_right)/(len(pred_wrong)+len(pred_right)+len(pred_none)),2)}% counting nulls\n")

        return pred_right, ts_right, pred_wrong, ts_wrong, pred_none, ts_none


    def assign_classification(self, i):
        high = (1 + self.TARGET_CHANGE)*self.klines['close'][i]
        low = (1 - self.TARGET_CHANGE)*self.klines['close'][i]


        i+=1

        for j in range(self.FUTURE_CHECK_LEN):
            try:
                is_high = self.klines['high'][i+j]>=high
                is_low = self.klines['low'][i+j]<=low

                if is_high:
                    if is_low:
                        return None
                    else:
                        return 1
                elif is_low:
                    return 0
            except:
                return None
        return None


    def save_ready_preds(self):
        print("Saving ready preds...")

        try:
            os.makedirs(self.READY_PRED_PATH[:-41])

        except  FileExistsError:
            pass

        pickle_out = open(self.READY_PRED_PATH, "wb")
        pickle.dump(self.pred_df, pickle_out)
        pickle_out.close()
        print("...saved\n")
            