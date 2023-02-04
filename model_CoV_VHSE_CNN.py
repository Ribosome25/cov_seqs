# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:37:37 2021

@author: Ruibo


Regression VHSE CNN with known params.
And save the predictions and Conv layers embedding.

Keras model. Run on tf env.
seq len are all 1273

"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
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
from myToolbox.Metrics import octuple
#%%
def write_perform(mssg, file_name="Results_CoV_VHSE_CNN_Rsearch.csv"):
    string = "\n{}" + ",{}"*(len(mssg)-1)
    with open("./Outputs/HyperTune/" + file_name, 'a') as file:
        file.write(string.format(*mssg))

#%%
# config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 3})
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def parse_one_seq_VHSE(seq, vhse):
    result = []
    for each_char in seq:
        result.append(vhse.loc[each_char].values)
    return np.hstack(result)

def covert_VHSE(df):
    # df with col "Seqs"
    from tqdm import tqdm
    vhse = pd.read_csv("VHSE.csv", header=None, index_col=0)
    Nn = len(df)
    Mm = 1273 * 8
    seqs = df["Seqs"]
    final_data = np.zeros((Nn, Mm))
    for idx, seq in tqdm(seqs.iteritems()):
        final_data[idx] = parse_one_seq_VHSE(seq, vhse)
    return pd.DataFrame(final_data)

#%%
RANDOM_SEED = 20
SAMPLE_FRAC = 1

_, df = loaddata.load_VHSE()
# _, df = loaddata.load_VHSE_Expression()


# df = df.iloc[:100]  # for test run
assert (_.index == df.index).all()  # 证明MPI 是阻塞的结果，顺序是不变的。

df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
_ = _.reindex(df.index)
_ = _.values.reshape(len(df), -1, 8)

X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

# # Uncomment for VHSE -> regression directly. 
# for mdl in [RFR(n_jobs=-1, verbose=True, min_samples_leaf=3),
#             Ridge(), LR()]:
#     mdl.fit(X_train.reshape(len(X_train), -1), y_train['fitness'])
#     pred = mdl.predict(X_test.reshape(len(X_test), -1))
#     ref_pref = octuple(y_test['fitness'].values.ravel(), pred.ravel(), False)[:6]
#     write_perform(["VHSE,{},{},{},SEED{}".format(SAMPLE_FRAC, str(mdl).replace(",", ";"),
#                                                  len(_), RANDOM_SEED), *ref_pref])

del df
# raise
#%%

def grid_search_model():
    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
        # keras.Input(shape=1273*8, name='Input'),

        layers.Conv1D(
            input_shape=(1273, 8),
            filters=256,
            kernel_size=6,
            strides=3,
            padding="valid",
            data_format="channels_last",
            activation='relu',
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="Conv1"
        ),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(
            filters=128,
            kernel_size=6,
            strides=3,
            padding="valid",
            data_format="channels_last",
            activation='relu',
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="Conv2",
        ),
        layers.MaxPooling1D(pool_size=2),

        layers.Flatten(),
        layers.Dense(128, activation="relu",
                      kernel_initializer='he_normal', name="layer4"),
        layers.Dense(64, activation="relu",
                      kernel_initializer='he_normal', name="layer5"),
        layers.Dense(32, activation="relu",
                      kernel_initializer='he_normal', name="layer6"),
        layers.Dense(1)
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr=2e-3, beta_1=0.9),  # Optimizer
        # optimizer=keras.optimizers.SGD(lr=5e-4, momentum=0.5),
        # Loss function to minimize
        # loss=keras.losses.MeanSquaredError(),
        loss='mse',
        # List of metrics to monitor
        metrics=[keras.metrics.mse]
    )
    return model



# Hyper tuned.
def random_search_model():
    model = keras.Sequential(
        [
        # keras.Input(shape=1273*8, name='Input'),

        layers.Conv1D(
            input_shape=(1273, 8),
            filters=128,
            kernel_size=6,
            strides=2,
            padding="valid",
            data_format="channels_last",
            activation='relu',
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="Conv1"
        ),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(
            filters=128,
            kernel_size=8,
            strides=4,
            padding="valid",
            data_format="channels_last",
            activation='relu',
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="Conv2",
        ),
        layers.MaxPooling1D(pool_size=2),

        layers.Flatten(),
        layers.Dense(32, activation="relu",
                      kernel_initializer='he_normal', name="layer4"),
        layers.Dense(16, activation="relu",
                      kernel_initializer='he_normal', name="layer5"),
        # layers.Dense(32, activation="relu",
                      # kernel_initializer='he_normal', name="layer6"),
        layers.Dense(1)
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(lr=2e-3, beta_1=0.9),  # Optimizer
        # optimizer=keras.optimizers.SGD(lr=5e-4, momentum=0.5),
        # Loss function to minimize
        # loss=keras.losses.MeanSquaredError(),
        loss='mse',
        # List of metrics to monitor
        metrics=[keras.metrics.mse]
    )
    return model


#%%

model = grid_search_model()
model = random_search_model()

print("Fit model on training data")
history = model.fit(
    X_train,
    y_train['fitness'],
    batch_size=64,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_valid, y_valid['fitness'].values),
)


np.save("cnn_valid_results.npy", model.predict(X_valid))
predictions = model.predict(X_test)
np.save("cnn_results.npy", predictions)
print("predictions shape:", predictions.shape)
performance = octuple(y_test['fitness'].values.ravel(), predictions.ravel(), False)[:6]
print(performance)
valid_err = history.history['val_loss'][-1]

write_perform(["Keras VHSE CNN,{},{},SEED{}".format(SAMPLE_FRAC, len(_), RANDOM_SEED), valid_err, *performance])

def myprint(s):
    with open('./Outputs/HyperTune/Results_CoV_VHSE_CNN.txt','a') as f:
        print(s, file=f)

model.summary(print_fn=myprint)

#%%
del X_train, X_valid, X_test
intermediate_layer_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer('max_pooling1d_1').output)

embed = pd.read_parquet("data/fitness_embeddings/VHSE_embedding_matched.parquet", engine='fastparquet')
embed = embed.values.reshape(len(embed), -1, 8)

intermediate_output = intermediate_layer_model.predict(embed)
np.save("precomputed/cnn_embedding_orderedsameasmatched.npy", intermediate_output)  # 顺序是不变的。

embed = pd.read_parquet("data/fitness_embeddings/from_HPC_VHSE_embedding.parquet", engine='fastparquet')
embed = embed.values.reshape(len(embed), -1, 8)

intermediate_output = intermediate_layer_model.predict(embed)
np.save("precomputed/cnn_embedding_orderedsameasfromhpc.npy", intermediate_output)  # 顺序是不变的。
