"""

https://keras.io/guides/keras_tuner/getting_started/

As shown below, the hyperparameters are actual values. In fact, they are just functions returning actual values. 
For example, hp.Int() returns an int value. Therefore, you can put them into variables, for loops, or if conditions.
"""


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import loaddata
from tensorflow.keras import layers

from keras.layers.convolutional import Convolution1D, MaxPooling1D


def build_model(hp):
    model = keras.Sequential()
    model.add(
        Convolution1D(
            input_shape=(1273, 8),
            filters=hp.Int("n_filters_1", min_value=64, max_value=256, step=64),
            kernel_size=hp.Int("ker_size_1", min_value=4, max_value=10, step=2),
            strides=hp.Int("stride_1", min_value=2, max_value=6, step=2),
            padding="valid",
            data_format="channels_last",
            activation='relu',
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="Conv1"
            )
    )
    model.add(MaxPooling1D(pool_size=2))

    if hp.Boolean("2nd_conv_layer"):
        model.add(
            Convolution1D(
            filters=hp.Int("n_filters_2", min_value=64, max_value=256, step=64),
            kernel_size=hp.Int("ker_size_2", min_value=4, max_value=8, step=2),
            strides=hp.Int("stride_2", min_value=2, max_value=6, step=2),
                padding="valid",
                data_format="channels_last",
                activation='relu',
                use_bias=True,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                name="Conv2",
                )
        )
    model.add(MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())

    if hp.Boolean("1st_Dense_layer"):
        model.add(
            layers.Dense(
                units=hp.Int("units_1", min_value=128, max_value=1024, step=128),
                activation="relu",
                kernel_initializer='glorot_uniform', name="layer1"
            )
        )

    if hp.Boolean("2nd_Dense_layer"):
        model.add(
            layers.Dense(
                units=hp.Int("units_2", min_value=64, max_value=512, step=64),
                activation="relu",
                kernel_initializer='glorot_uniform', name="layer2"
            )
        )

    if hp.Boolean("3rd_Dense_layer"):
        model.add(
            layers.Dense(
                units=hp.Int("units_3", min_value=32, max_value=256, step=32),
                activation="relu",
                kernel_initializer='glorot_uniform', name="layer3"
            )
        )

    if hp.Boolean("4th_Dense_layer"):
        model.add(
            layers.Dense(
                units=hp.Int("units_4", min_value=16, max_value=64, step=16),
                activation="relu",
                kernel_initializer='glorot_uniform', name="layer4"
            )
        )

    model.add(layers.Dense(1, activation="sigmoid"))

    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse"],
    )
    return model

import keras_tuner as kt

# build_model(kt.HyperParameters())

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="mse",
    max_trials=20,
    executions_per_trial=10,
    overwrite=False,
    directory="Outputs/HyperTune",
    project_name="cnn_2",
)
print("==================Search space summary====================")
print(tuner.search_space_summary())

#%%
def data():
    '''
    Data providing function:
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    SAMPLE_FRAC = 1
    RANDOM_SEED = 20
    
    _, df = loaddata.load_VHSE()
    # _, df = loaddata.load_VHSE_Expression()
    assert (_.index == df.index).all()  # 证明MPI 是阻塞的结果，顺序是不变的。


    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    _ = _.reindex(df.index)
    _ = _.values.reshape(len(df), -1, 8)

    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

    y_train = y_train['fitness'].values
    y_valid = y_valid['fitness'].values

    return X_train, y_train, X_valid, y_valid

#%%
x_train, y_train, x_val, y_val = data()
print("==================Data loaded====================")

tuner.search(x_train, y_train, epochs=30, validation_data=(x_val, y_val))

#%% Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 1273, 8))
print("==================Best model summary====================")

print(best_model.summary())
print("==================Results summary====================")

print(tuner.results_summary())
