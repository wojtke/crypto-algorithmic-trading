import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
import time
import os
import numpy as np
from datetime import datetime

from utils import FilenameParser

# SETTINGS
EPOCHS = 15
BATCH_SIZE = 32
NODES = 96
DENSE = 20

dataset_name = '15m-ehh-24.08.20'

file_list = FilenameParser.get_file_list(dataset_name)

for filename in file_list:

    with open(f'TRAIN_DATA/{dataset_name}/{filename}-t.pickle', 'rb') as pickle_in:
        train_x, train_y = pickle.load(pickle_in)
    with open(f'TRAIN_DATA/{dataset_name}/{filename}-v.pickle', 'rb') as pickle_in:
        validation_x, validation_y = pickle.load(pickle_in)

    
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    validation_x = np.asarray(validation_x)
    validation_y = np.asarray(validation_y)
    

    NAME = f"{filename}-{datetime.now().strftime('%d%b%y-%H.%M.%S')}" 
    print(f"Teraz robie sobie {NAME}")

    tensorboard = TensorBoard(log_dir=f"TB_LOG\\{dataset_name}\\{NAME}")

    model = Sequential()

    model.add(LSTM(NODES, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(NODES, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(LSTM(NODES, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(DENSE, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))


    opt = tf.keras.optimizers.Adam(lr=0.00008)

    model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
                  )

    #zapisywanko modelu
    checkpoint = ModelCheckpoint(filepath=f"MODELS/{dataset_name}/{NAME}/"+"{epoch:02d}-TL{loss:.3f}-TA{accuracy:.3f}_VL{val_loss:.3f}-VA{val_accuracy:.3f}.model", 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=False, 
            mode='min')
    try:
        os.makedirs(f"MODELS/{dataset_name}/{NAME}")
    except  FileExistsError:
        pass

    
    #szybkie zatrzymywanie
    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
            restore_best_weights=True)
    
    #proste zapisanie w txt ustawien, tak zeby mi bylo wygodnie je na szybko przeczytac
    f = open(f"MODELS/{dataset_name}/{NAME}/details.txt", "w")
    f.write(f"EPOCHS {EPOCHS}\n"+ 
            f"BATCH_SIZE {BATCH_SIZE}\n"+
            f"NODES {NODES}\n"+
            f"DENSE {DENSE}\n")
    f.close()

    history = model.fit(
            train_x, train_y,
            shuffle=True,
            epochs=EPOCHS,
            verbose=2,
            batch_size=BATCH_SIZE,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint, early_stopping],
            )

    