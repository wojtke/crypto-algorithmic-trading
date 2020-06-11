import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import time
import filename_parser
import os
import numpy as np
from datetime import datetime

EPOCHS = 30 
BATCH_SIZE = 32
NODES = 64
DENSE = 16

dataset_name = '11.06.20_shl'

file_list = filename_parser.get_file_list(dataset_name)

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

    tensorboard = TensorBoard(log_dir=f"tensorboard_logs\\{dataset_name}\\{NAME}")

    model = Sequential()

    model.add(LSTM(NODES, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(LSTM(NODES, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(NODES))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(DENSE, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(2, activation='softmax'))


    opt = tf.keras.optimizers.RMSprop(lr=0.0001)

    model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
                  )


    checkpoint = ModelCheckpoint(filepath=f"models/{dataset_name}/{NAME}/{NAME}"+"-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}.model", 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=False, 
            mode='min')
    try:#
        os.makedirs(f"models/{dataset_name}/{NAME}")

    except  FileExistsError:
        pass

    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            restore_best_weights=True)

    history = model.fit(
            train_x, train_y,
            shuffle=True,
            epochs=EPOCHS,
            verbose=2,
            batch_size=BATCH_SIZE,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint, early_stopping],
            )