# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:15:39 2021

@author: RUIBZHAN
"""
RANDOM_SEED = 20
SAMPLE_FRAC = 1

def write_perform(mssg, file_name="Results_CoV_VHSE_CNN.csv"):
    string = "\n{}" + ",{}"*(len(mssg)-1)
    with open("./Outputs/CoVClf/" + file_name, 'a') as file:
        file.write(string.format(*mssg))


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from model_CoV_seq_clf import convert_to_clf_data
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import loaddata
from topo_reg import calculate_distance, reconstruct
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from myToolbox.Metrics import sextuple, twocat_sextuple_1d

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

#%%

_, df = loaddata.load_VHSE()
# _, df = loaddata.load_VHSE_Expression()
assert (_.index == df.index).all()  # 证明MPI 是阻塞的结果，顺序是不变的。

df = convert_to_clf_data(df)


df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
_ = _.reindex(df.index)
_ = _.values.reshape(len(df), -1, 8)

X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

y_tr = np.hstack((1 - y_train['fitness'].values.reshape(-1, 1), 
            y_train['fitness'].values.reshape(-1, 1))).astype('float32')
y_val = np.hstack((1 - y_valid['fitness'].values.reshape(-1, 1), 
            y_valid['fitness'].values.reshape(-1, 1))).astype('float32')




def run(n_chnls=(256, 128), kers_strides=(12, 6, 10, 5), opt='adam', layer_config='1111', lr=1e-3):
    # 第二层用n_channels 第二位来控制。

    model = keras.Sequential()
    model.add(
        layers.Conv1D(
            input_shape=(1273, 8),
            filters=n_chnls[0],
            kernel_size=kers_strides[0],
            strides=kers_strides[1],
            padding="valid",
            data_format="channels_last",
            activation='relu',
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="Conv1"
            )
        )

    model.add(layers.MaxPooling1D(pool_size=2))

    if n_chnls[1] > 0:
        model.add(
            layers.Conv1D(
                filters=n_chnls[1],
                kernel_size=kers_strides[2],
                strides=kers_strides[3],
                padding="valid",
                data_format="channels_last",
                activation='relu',
                use_bias=True,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                name="Conv2",
                )
            )
        model.add(layers.MaxPooling1D(pool_size=2))
    
    model.add(layers.Flatten())
    
    if layer_config[0]:
        model.add(layers.Dense(256, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer1"))
    if layer_config[1]:
        model.add(layers.Dense(128, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer2"))
    if layer_config[2]:
        model.add(layers.Dense(64, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer3"))
    if layer_config[3]:
        model.add(layers.Dense(32, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer4"))        

    model.add(layers.Dense(2, activation="softmax"))

    if opt == 'adam':
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9)
    elif opt == 'sgd':
        optimizer=keras.optimizers.SGD(lr=lr, momentum=0.5)
    model.compile(
        optimizer=optimizer,  # Optimizer
        loss="categorical_crossentropy",
        # List of metrics to monitor
        metrics=["AUC"]
        )
    print("Fit model on training data")
    history = model.fit(
        X_train,
        y_tr,
        batch_size=64,
        epochs=200,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_valid, y_val),
    )
    valid_err = history.history['val_loss'][-1]
    if np.isnan(valid_err):
        write_perform(["Keras VHSE CNN Model,Size{},{},{},{},{},{}".format(
            SAMPLE_FRAC, n_chnls, kers_strides, opt, layer_config, lr),
                   valid_err])
        return None
    predictions = model.predict(X_test)
    print("predictions shape:", predictions.shape)
    performance = twocat_sextuple_1d(y_test['fitness'].values, predictions)
    print(performance)
    write_perform(["Keras VHSE CNN Model,Size{},{},{},{},{},{}".format(
        SAMPLE_FRAC, n_chnls, kers_strides, opt, layer_config, lr),
                   valid_err, *performance])


for channel in [(128, 128), (256, 256), (128, 256), (256, 128), (128,0), (256,0)]:
# for channel in [(128, 256), (256, 128), (256,0)]:
    for kers in [(12,6,10,5), (14,7,10,5), (10,5,8,4), (8,4,6,3), (6,3,6,3), (3,3,2,2)]:
    # for kers in [(8,4,6,3), (6,3,6,3)]:
        for opt in ['adam']:
            for layer_config in ["1111", "0111"]:#, '0011', '0110']:
                for lr in [1e-3]:
                    run(channel, kers, opt, layer_config, lr)