import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
import pickle
from collections import deque
from utils import Progressbar, create_dir
import os
from sklearn import preprocessing
import random

from vars import Vars

class Preprocessor:
    def __init__(self, SYMBOL=None, INTERVAL=None, TEST_PERIOD=None, market='spot'):
        self.SYMBOL = SYMBOL or Vars.SYMBOL
        self.INTERVAL = INTERVAL or Vars.INTERVAL
        self.TEST_PERIOD = TEST_PERIOD or Vars.TEST_PERIOD

        self.klines_path = Vars.main_path + f'RAW_DATA/Binance_{market}_{self.SYMBOL}USDT_{self.INTERVAL}.json' 

        self.klines = pd.DataFrame()


    def repreprocess(self, MODEL, do_not_use_ready=False, save=True):
        self.read_model_details(MODEL)

        if do_not_use_ready:
            print("It was specified not to use ready predictions\n")
        else:
            try:
                self.pred_df = pickle.load(open( self.READY_PRED_PATH, "rb" ))
                print("Ready saved predictions found!")
                return self.pred_df

            except:
                print("Ready saved predictions not found.")

        print("Repreprocessing...")
        if self.klines.empty:
            self.klines_load()

        df = self.data_to_pct(self.klines.copy())

        df = self.data_scale(df)
        seqs, ts, target, pred_index = self.create_seqs(df, return_more=True)

        preds = self.data_predict(seqs)

        predx = pd.DataFrame(preds)[1]

        self.pred_df = pd.DataFrame(data={"preds":(np.array(predx)), "ts":ts, "target":target}, index=pred_index)

        if save:
            self.save_ready_preds()

        print("...repreprocessed and saved")

    
    def preprocess(self, TARGET_CHANGE, PAST_SEQ_LEN, FUTURE_CHECK_LEN, SPECIAL_NAME, date_str):
        self.TARGET_CHANGE = TARGET_CHANGE
        self.PAST_SEQ_LEN = PAST_SEQ_LEN
        self.FUTURE_CHECK_LEN = FUTURE_CHECK_LEN
        self.SPECIAL_NAME = SPECIAL_NAME

        self.SAVE_NAME = f'{self.SYMBOL}USDT{self.INTERVAL}-{PAST_SEQ_LEN}x{FUTURE_CHECK_LEN}~{TARGET_CHANGE}'
        
        self.date_str = date_str

        print(f"\n ----- Preprocessing {self.SAVE_NAME} ----- ")

        if self.klines.empty:
            self.klines_load()

        df = self.data_to_pct(self.klines.copy())

        df_t, df_v = self.preprocess_separate(df)
        del df

        df_t = self.data_scale(df_t, from_saved_scaler=False)
        df_v = self.data_scale(df_v, from_saved_scaler=True)

        train_x, train_y = self.create_seqs(df_t)
        del df_t

        val_x, val_y = self.create_seqs(df_v)
        del df_v

        train_x, train_y = self.balance_and_shuffle(train_x, train_y)
        val_x, val_y = self.balance_and_shuffle(val_x, val_y)

        print('--------------------------------------')
        print(f"Train data len: {len(train_y)}, Validation data len: {len(val_y)}")
        print(f"TRAIN:   Sells: {np.count_nonzero(train_y==0)}, Buys: {np.count_nonzero(train_y==1)}")
        print(f"TEST:    Sells: {np.count_nonzero(val_y==0)}, Buys: {np.count_nonzero(val_y==1)}")
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


    def klines_load(self, load_test_period=False):
        print("Loading klines...")
        print(self.klines_path)

        klines_uncut = pd.read_json(self.klines_path)
        klines = pd.DataFrame()

        if load_test_period:
            ts1 = 1000*datetime.timestamp(self.TEST_PERIOD[0])
            ts2 = 1000*datetime.timestamp(self.TEST_PERIOD[1])

            a = np.array(klines_uncut.loc[klines_uncut[0].isin([ts1, ts2])].index)

            klines = klines_uncut.iloc[a[0]:a[1]]
            klines = klines.append(klines_uncut.iloc[a[0]:a[1]])
        else:
            klines = klines_uncut


        klines = klines[[0,1,2,3,4,5]]
        klines.columns = ['openTimestamp', 'open', 'high', 'low', 'close', 'volume']

        
        for c in klines.columns:
            klines[c] = pd.to_numeric(klines[c], errors='coerce')

        self.klines = klines
        print(f"...klines loaded ({len(klines)})\n")
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

    def create_seqs(self, df, return_more=False):
        print("Creating sequences...")

        sliding_window = deque(maxlen=self.PAST_SEQ_LEN) 

        seqs=[]
        ts =[]
        target=[]
        pred_index=[] 

        pb = Progressbar(len(df.index), name="Sequentialisation")

        for i in range(len(df.index)-1): 

            sliding_window.append(df.values[i][1:]) 

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
        if return_more:
            return np.array(seqs), ts, target, pred_index
        else:
            return np.array(seqs), np.array(target)

    def balance_and_shuffle(self, X, Y):
        print("Balancing and shuffling...")

        indices = np.nonzero(Y==None)[0]

        print(f"Deleting {len(indices)} Nones")

        X = np.delete(X, indices, axis=0)
        Y = np.delete(Y, indices)

        buy_count = np.count_nonzero(Y==1)
        sell_count = np.count_nonzero(Y==0)


        pct_buys = round(100 * buy_count / len(Y))
        print(f'({buy_count} buys, {sell_count} sells - {pct_buys}%/{100 - pct_buys}%)')

        if buy_count>sell_count:
            indices = np.nonzero(Y==1)[0]
            dif = buy_count-sell_count

        else:
            indices = np.nonzero(Y==0)[0]
            dif = sell_count-buy_count

        np.random.shuffle(indices)
        indices = indices[:dif]

        X = np.delete(X, indices, axis=0)
        Y = np.delete(Y, indices)

        print(f'Equalized both buys/sells to {min(buy_count, sell_count)}')

        indices = np.arange(len(Y))
        np.random.shuffle(indices)

        X = X[indices]
        Y = Y[indices]

        print(f"...balancing and shuffling done ({len(Y)})\n")
        return X, Y


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

        ts1 = int(1000*datetime.timestamp(self.TEST_PERIOD[0]))
        ts2 = 1000*datetime.timestamp(self.TEST_PERIOD[1])
        a = np.array(self.klines.loc[self.klines['openTimestamp'].isin([ts1, ts2])].index)

        train_mask = (df['openTimestamp']<ts1) | (df['openTimestamp']>=ts2)

        df_t = df.loc[train_mask]
        df_v = df.loc[~train_mask]

        print(f"Train set period: {self.TEST_PERIOD[0]} - {self.TEST_PERIOD[1]}")

        print(f"Train set candles: {len(df_t.index)}")
        print(f"Validation set candles: {len(df_v.index)}")
        print(f"Val: {round(100 * len(df_v.index) / len(df), 2)} %")

        print("...separating done\n")
        return df_t, df_v


    def save_train_and_val(self, train_x, train_y, val_x, val_y):
        print('\nSaving training and validation data...')

        create_dir(f'TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}')

        pickle_out = open(f"TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}/{self.SAVE_NAME}-t.pickle", "wb")
        pickle.dump((train_x, train_y), pickle_out)
        pickle_out.close()

        pickle_out = open(f"TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}/{self.SAVE_NAME}-v.pickle", "wb")
        pickle.dump((val_x, val_y), pickle_out)
        pickle_out.close()

        print('...training data saved as: ')
        print(f'TRAIN_DATA/{self.INTERVAL}-{self.SPECIAL_NAME}-{self.date_str}/{self.SAVE_NAME}')
    