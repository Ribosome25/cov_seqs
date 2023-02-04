# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:21:42 2021

@author: ruibzhan
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import loaddata
from myToolbox.Metrics import octuple

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

def write_perform(mssg, file_name="Results_FC_test.csv"):
    string = "\n{}" + ",{}"*(len(mssg)-1)
    with open("./Outputs/HyperTune" + file_name, 'a') as file:
    # with open("C:/Users/ruibzhan/OneDrive - Texas Tech University/New Project/results" + file_name, 'a') as file:
        file.write(string.format(*mssg))

# Define Sequential model with 3 layers
def run(data='sci', opt='adam', layer_config='11111', lr=1e-3, d_frac=0.05):
    
    model = keras.Sequential()
    layer_config = [int(x) for x in layer_config]
    
    model.add(keras.Input(shape=1024))
    if layer_config[0]:
        model.add(layers.Dense(1024, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer0"))
    if layer_config[1]:
        model.add(layers.Dense(512, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer1"))
    if layer_config[2]:
        model.add(layers.Dense(256, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer2"))
    if layer_config[3]:
        model.add(layers.Dense(128, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer3"))
    if layer_config[4]:
        model.add(layers.Dense(64, activation="relu",
                       kernel_initializer='glorot_uniform', name="layer4"))
    if len(layer_config) > 5:  # new
        if layer_config[5]:
            model.add(layers.Dense(32, activation="relu",
                           kernel_initializer='glorot_uniform', name="layer5"))
        if layer_config[6]:
            model.add(layers.Dense(16, activation="relu",
                           kernel_initializer='glorot_uniform', name="layer6"))

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
    #%%
    RANDOM_SEED = 20
    
    if data == 'sci':
        # _, df = loaddata.load_Sci_embeddings(dataset='expression')
        _, df = loaddata.load_Sci_embeddings(dataset='fitness')
    elif data == 'bert':
        # _, df = loaddata.load_Bert_embeddings('avg', dataset='expression')
        _, df = loaddata.load_Bert_embeddings('avg', dataset='fitness')
    
    redo_index = ["I{}".format(x) for x in range(len(df))]
    _.index = redo_index
    df.index = redo_index
    
    # ---- For testing ----
    df = df.sample(frac=d_frac, random_state=RANDOM_SEED)
    _ = _.reindex(df.index)
    
    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

    #%%
    print("Fit model on training data")
    if data == 'sci':
        b = 64
    else:
        b = 63
    history = model.fit(
        X_train,
        y_train['fitness'],
        batch_size=b,
        epochs=200,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_valid.values, y_valid['fitness'].values),
    )
    valid_err = history.history['val_loss'][-1]
    predictions = model.predict(X_test)
    print("predictions shape:", predictions.shape)
    performance = octuple(y_test['fitness'].values.ravel(), predictions.ravel(), False)[:6]
    print(performance)
    write_perform(["Keras FC Model {},{},{},{},{}".format(d_frac, data, opt, layer_config, lr),
                   valid_err, *performance])

# Hyper tuned. 
for d_frac in [1]:
    for data in ['sci']:
        for opt in ['adam']:
            for layer_config in ["0001111"]:
                for lr in [1.2e-3]:
                    try:
                        run(data, opt, layer_config, lr, d_frac)
                    except ValueError:
                        continue
raise ValueError("Success.")

for d_frac in [0.005, 0.05, 1]:
    for data in ['sci', 'bert']:
        for opt in ['sgd', 'adam']:
            for layer_config in ["11111", "01111", '00111', '00011']:
                for lr in [1e-2, 1e-3, 1e-4]:
                    try:
                        run(data, opt, layer_config, lr, d_frac)
                    except ValueError:
                        continue
