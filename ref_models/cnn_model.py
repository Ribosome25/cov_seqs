"""
Build CNN models for VHSE, etc.  

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


def build_VHSE_CNN(max_length=1273, n_chnls=(256, 128), kers_strides=(12, 6, 10, 5), opt='adam', layer_config='1111', lr=1e-3):
    """
    第二层Conv 用n_channels 第二位来控制。 如果第二位是0，只用1层Conv layer。
    layer_config 是FC layers 的config。1 表示有，0表示没有。
    FC layers 固定每层的node 数。256-128-64-32，用0 和 1 表示采用/不采用
    """

    model = keras.Sequential()
    model.add(
        layers.Conv1D(
            input_shape=(max_length, 8),
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
    model.add(layers.Dense(1))

    if opt == 'adam':
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9)
    elif opt == 'sgd':
        optimizer=keras.optimizers.SGD(lr=lr, momentum=0.5)
    model.compile(
        optimizer=optimizer,  # Optimizer
        loss='mse',
        # List of metrics to monitor
        metrics=[keras.metrics.mse]
        )
    
    return model