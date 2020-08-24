import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import pickle
from collections import deque
from utils import Progressbar
import os
from sklearn import preprocessing
import random

from vars import Vars

class Preprocessor:
    def __init__(self, SYMBOL=None, INTERVAL=None, WINDOWS=None, klines='spot'):
        if not SYMBOL:
            self.SYMBOL = Vars.SYMBOL
        else:
            self.SYMBOL = SYMBOL

        if not INTERVAL:
            self.INTERVAL = Vars.INTERVAL
        else:
            self.INTERVAL = INTERVAL

        if not WINDOWS:
            self.WINDOWS = Vars.WINDOWS
        else:
            self.WINDOWS = WINDOWS

        if klines == 'spot':
            self.klines_path = Vars.main_path + f'RAW_DATA/Binance_{self.SYMBOL}USDT_{self.INTERVAL}.json' 
        elif klines == 'futures':
            self.klines_path = Vars.main_path + f'RAW_DATA/Binance_futures_{self.SYMBOL}USDT_{self.INTERVAL}.json' 

        self.klines = pd.DataFrame()


    def repreprocess(self, MODEL, do_not_use_ready=False, save=True):
        self.read_model_details(MODEL)

        try:
            self.pred_df = pickle.load(open( self.READY_PRED_PATH, "rb" ))
            print("Ready saved predictions found!")

            if do_not_use_ready:
                print("It was specified not to use ready predictions\n")
                raise Exception("do_not_use_ready=True")

        except:
            print("Repreprocessing shit...")
            if self.klines.empty:

                self.klines_load()

            df = self.data_to_pct(self.klines.copy())

            df = self.data_scale(df)
            seqs, ts, target, pred_index = self.data_make_sequences(df)

            preds = self.data_predict(seqs)

            predx = pd.DataFrame(preds)[1]

            self.pred_df = pd.DataFrame(data={"preds":(np.array(predx)), "ts":ts, "target":target}, index=pred_index)

            if save:
                self.save_ready_preds()

            print("...shit repreprocessed and saved")

    
    def preprocess(self, TARGET_CHANGE, PAST_SEQ_LEN, FUTURE_CHECK_LEN, SPECIAL_NAME, date_str):

        self.TARGET_CHANGE = TARGET_CHANGE
        self.PAST_SEQ_LEN = PAST_SEQ_LEN
        self.FUTURE_CHECK_LEN = FUTURE_CHECK_LEN
        self.SPECIAL_NAME = SPECIAL_NAME

        self.SAVE_NAME = f'{self.SYMBOL}USDT{self.INTERVAL}-{PAST_SEQ_LEN}x{FUTURE_CHECK_LEN}~{TARGET_CHANGE}'
        
        self.date_str = date_str

        print(f"\n ----- Preprocessing {self.SAVE_NAME} ----- ")

        self.klines_load(load_all=True)

        df = self.data_to_pct(self.klines.copy())

        df_t, df_v = self.preprocess_separate(df)
        del df

        df_t = self.data_scale(df_t, from_saved_scaler=False)
        df_v = self.data_scale(df_v, from_saved_scaler=True)

        train_x, train_y = self.preprocess_create_seqs(df_t, "train")
        del df_t

        val_x, val_y = self.preprocess_create_seqs(df_v, 'val')
        del df_v

        print('--------------------------------------')
        print(f"Train data len: {len(train_y)}, Validation data len: {len(val_y)}")
        print(f"TRAIN:   Sells: {train_y.count(0)}, Buys: {train_y.count(1)}")
        print(f"TEST:    Sells: {val_y.count(0)}, Buys: {val_y.count(1)}")
        print(f"Validation is {round(100 * len(val_y) / (len(val_y) + len(train_y)), 2)}% of all data")

        self.save_train_and_val(train_x, train_y, val_x, val_y)
    

    def read_model_details(self, MODEL):
        self.MODEL_PATH = Vars.main_path + "MODELS/" + MODEL
        self.SCALER_PATH = Vars.main_path + "SCALERS/" + MODEL[:-58] + '-scaler_data.pickle'
        self.READY_PRED_PATH = Vars.main_path + "READY_PRED/" + MODEL[:-5] + 'pickle'

        ix = MODEL.find('~')+1
        self.TARGET_CHANGE = float(MODEL[ix:MODEL.find('-', ix)])

        self.FUTURE_CHECK_LEN = int(MODEL[MODEL.find('x')+1:ix-1])

        ix=MODEL[MODEL.find('x')-1::-1].find('-')
        self.PAST_SEQ_LEN = int(MODEL[MODEL.find('x')-ix:MODEL.find('x')])

        self.SYMBOL = MODEL[MODEL.find('/')+1:MODEL.find('USDT')]

        self.INTERVAL = MODEL[MODEL.find('USDT')+4:MODEL.find('x')-ix-1]


    def klines_load(self, load_all=False):
        print("Loading klines...")
        print(self.klines_path)

        klines_uncut = pd.read_json(self.klines_path)
        klines = pd.DataFrame()

        if not load_all:
            for w in self.WINDOWS: 
                ts1 = 1000*datetime.timestamp(w[0] - w[1])
                ts2 = 1000*datetime.timestamp(w[0] + w[1])
                a = np.array(klines_uncut.loc[klines_uncut[0].isin([ts1, ts2])].index)

                if klines.empty:
                    klines = klines_uncut.iloc[a[0]:a[1]]
                else:
                    klines = klines.append(klines_uncut.iloc[a[0]:a[1]])
        else:
            klines = klines_uncut


        klines = klines[[0,1,2,3,4,5]]
        klines.columns = ['openTimestamp', 'open', 'high', 'low', 'close', 'volume']

        for c in klines.columns:
            klines[c] = pd.to_numeric(klines[c], errors='coerce')
        '''
        df = df.reset_index()
        df = df.drop(columns=["index"])
        '''
        self.klines = klines
        print("...klines loaded\n")
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


    def data_scale(self, df, from_saved_scaler=True):
        print("Scaling data...")

        if from_saved_scaler:
            pickle_scaler_data = pickle.load( open( self.SCALER_PATH, "rb" ) )

            mean = pickle_scaler_data[0]
            scale =pickle_scaler_data[1]

        else:
            scaler = preprocessing.StandardScaler().fit(df.values)

            mean = scaler.mean_
            scale = scaler.scale_

            try:
                os.makedirs(f'SCALERS/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}')

            except  FileExistsError:
                pass

            self.SCALER_PATH = f"SCALERS/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}/{self.SAVE_NAME}-scaler_data.pickle"

            pickle_out = open(self.SCALER_PATH, "wb")
            pickle.dump([scaler.mean_, scaler.scale_], pickle_out)
            pickle_out.close()
            print(f'Saved the scaler')


        print("Scaler means: ", mean)
        print("Scaler scales: ", scale)
        i=0
        for col in df.columns: 
            if col != "openTimestamp":
                df[col] = df[col].sub(mean[i]).div(scale[i])
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
        

    def preprocess_separate(self, df):
        print("Separating data...")

        df_t = pd.DataFrame()
        df_v = pd.DataFrame(data={
            "openTimestamp": [1337]})  # this is passed just for this df to have an index of 0, this row is later dropped

        for w in self.WINDOWS:
            ts1 = 1000*datetime.timestamp(w[0] - w[1])
            ts2 = 1000*datetime.timestamp(w[0] + w[1])
            a = np.array(self.klines.loc[self.klines['openTimestamp'].isin([ts1, ts2])].index)

            # parts of train set and val set overlap by 1 row, should not be a major leak problem,
            # but if it is, it needs to be fixed
            df_t = df_t.append(df.iloc[df_v.index[-1]:a[0] + 1])
            df_v = df_v.append(df.iloc[a[0]:a[1] + 1])

            print(datetime.fromtimestamp(df['openTimestamp'][a[0]] / 1000), "-",
                  datetime.fromtimestamp(df['openTimestamp'][a[1]] / 1000))
            print(f"Val window of {a[1] - a[0]} candles")

        df_t = df_t.append(df.iloc[df_v.index[-1]:df.index[-1]])
        df_v = df_v.drop([0])

        print(f"\nTrain set candles: {len(df_t.index)}")
        print(f"Validation set candles: {len(df_v.index)}")
        print(f"Val: {round(100 * len(df_v.index) / len(df), 2)} %")

        # droppin and cleaning
        df_t.drop(columns=['openTimestamp'], inplace=True)
        df_v.drop(columns=['openTimestamp'], inplace=True)

        print("...separating done\n")
        return df_t, df_v


    def preprocess_create_seqs(self, df, which):
        print(f"Creating {which} sequences...")

        sliding_window = deque(maxlen=self.PAST_SEQ_LEN)

        # it is easier to split between two to balance out later
        buys = []
        sells = []

        pb = Progressbar(len(df.index), name="Sequentialisation")

        for i in range(len(df.index)-1): 

            sliding_window.append(df.values[i]) #adds single row of everything 

            if len(sliding_window) == self.PAST_SEQ_LEN: #when sliding window is of desired length
                classification = self.assign_classification(df.index[i])
                if classification == 0:
                    sells.append([np.array(sliding_window), classification])
                elif classification == 1:
                    buys.append([np.array(sliding_window), classification])

            if df.index[i]+1!=df.index[i+1]:
                sliding_window.clear()

            pb.update(i)
        del pb
 
        random.shuffle(buys)
        random.shuffle(sells)

        pct_buys = round(100 * len(buys) / (len(buys) + len(sells)))
        print(f'Created {which} seqs, ({len(buys)} buys, {len(sells)} sells - {pct_buys}%/{100 - pct_buys}%)')

        lower = min(len(buys), len(sells))

        buys = buys[:lower]
        sells = sells[:lower]

        print(f'Equalized both {which} buys/sells to {lower}')

        sequential_data = buys + sells
        del buys
        del sells

        random.shuffle(sequential_data)  # shuffle again

        X = []
        y = []
        for seq, target in sequential_data:
            X.append(seq)
            y.append(target)

        print(f'Returning {which} sequences ({len(y)})')

        print(f"...creating {which} sequences done\n")
        return np.array(X), y


    def save_train_and_val(self, train_x, train_y, val_x, val_y):
        print('\nSaving training and validation data...')

        try:
            os.makedirs(f'TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}')

        except  FileExistsError:
            pass

        pickle_out = open(f"TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}/{self.SAVE_NAME}-t.pickle", "wb")
        pickle.dump((train_x, train_y), pickle_out)
        pickle_out.close()

        pickle_out = open(f"TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}/{self.SAVE_NAME}-v.pickle", "wb")
        pickle.dump((val_x, val_y), pickle_out)
        pickle_out.close()

        print('...training data saved as: ')
        print(f'TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}/{self.SAVE_NAME}')
    