# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:37:37 2021

@author: Ruibo

Keras model. Run on tf env.

seq 长度都是1273

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
from sklearn.base import BaseEstimator

from model_CoV_seq_clf import convert_to_clf_data

from myToolbox.Metrics import sextuple, twocat_sextuple_1d
#%%
def write_perform(mssg, file_name="Results_CoV_VHSE_CNN.csv"):
    string = "\n{}" + ",{}"*(len(mssg)-1)
    with open("./Outputs/CoVClf/" + file_name, 'a') as file:
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
def VHSE_topo(n_anchors=500, mdl=LR(), rbf_gamma=1, knn=5):

    if isinstance(mdl, BaseEstimator):
        mdl = [mdl]
    if isinstance(knn, (int, float, complex)):
        knn = [knn]
    if isinstance(rbf_gamma, (int, float, complex)):
        rbf_gamma = [rbf_gamma]
    if knn is None:
        knn = []
    if rbf_gamma is None:
        rbf_gamma = []

    _, df = loaddata.load_VHSE()
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    _ = _.reindex(df.index)
    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)
    anchors_idx = calculate_distance.select_anchor_idx(y_train, y_train, n_anchors)
    if 0:
        dist_all = pd.DataFrame(
                pairwise_distances(_, _.loc[anchors_idx], metric='cityblock'),
                index=df.index)
    else:
        embed = _.values.reshape(len(_), -1, 8)
        # ii_df = df_raw.index.get_indexer(df.index)
        ii_anchors_embed = df.index.get_indexer(anchors_idx)
        ii_anchors_train = y_train.index.get_indexer(anchors_idx)

        dist_all = np.zeros((len(df), len(anchors_idx)), dtype=np.float32)
        for ii in range(embed.shape[1]):  # sum of 8维空间上的Euclid
            print(ii)
            dist_all += pairwise_distances(embed[:, ii, :], embed[ii_anchors_embed, ii, :])
        dist_all = pd.DataFrame(dist_all, index=df.index, columns=anchors_idx)

    dist_all.columns = anchors_idx
    dist_train = dist_all.loc[y_train.index]
    dist_valid = dist_all.loc[y_valid.index]
    dist_test = dist_all.loc[y_test.index]

    ii_anchors = y_train.index.get_indexer(anchors_idx)
    dist_y_train = pairwise_distances(y_train["fitness"].values.reshape(-1, 1),
                              y_train["fitness"].values.reshape(-1, 1)[ii_anchors, :])
    
    metric = 'Sum Euclid'
    submtr = 'vhse'

    v_fname = 'Results_CNN_topo_validset.csv'
    t_fname = "Results_CNN_topo_testset.csv"

    #%% modelling of distances
    for each_mdl in mdl:
        each_mdl.fit(dist_train.values, dist_y_train)
        dist_array_valid = each_mdl.predict(dist_valid.values).T
        dist_array_test = each_mdl.predict(dist_test.values).T
        # 对 LR 正反距离都一样
        for each_gamma in rbf_gamma:
            response_array_r_v = reconstruct.rbf(dist_array_valid, y_train["fitness"], anchors_idx, each_gamma, False)
            # Mod here:
            response_array_r_v = np.hstack((1 - response_array_r_v, response_array_r_v))
            performance_r_v = twocat_sextuple_1d(y_valid['fitness'], response_array_r_v)

            print("Valid: RBF: Accuracy {}, Precision {}, Recall {}, F1-score {}, ROC AUC {}, PRC AUC {}".format(*performance_r_v))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "rbf {}".format(each_gamma),
                           *performance_r_v], v_fname)
        #     # np.save("topo_valid_result.npy", response_array_r_v)  #TODO: temporary
        #     # np.save("actual_valid.npy", y_valid['fitness'].values)

            response_array_r_t = reconstruct.rbf(dist_array_test, y_train["fitness"], anchors_idx, each_gamma, False)
            response_array_r_t = np.hstack((1 - response_array_r_t, response_array_r_t))
            performance_r_t = twocat_sextuple_1d(y_test['fitness'], response_array_r_t)

            print("Test: RBF: Accuracy {}, Precision {}, Recall {}, F1-score {}, ROC AUC {}, PRC AUC {}".format(*performance_r_t))
            write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "rbf {}".format(each_gamma),
                           *performance_r_t], t_fname)
            # np.save("topo_result.npy", response_array_r_t)  #TODO: temporary
            # np.save("actual_result.npy", y_test['fitness'].values)

        for each_k in knn:
            response_array_k_v = reconstruct.knn(dist_array_valid, y_train["fitness"], anchors_idx, knn=each_k)
            response_array_k_v = np.hstack((1 - response_array_k_v, response_array_k_v))
            performance_k_v = twocat_sextuple_1d(y_valid['fitness'], response_array_k_v)

            print("Valid: RBF: Accuracy {}, Precision {}, Recall {}, F1-score {}, ROC AUC {}, PRC AUC {}".format(*performance_k_v))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_v], v_fname)

            response_array_k_t = reconstruct.knn(dist_array_test, y_train["fitness"], anchors_idx, knn=each_k)
            response_array_k_t = np.hstack((1 - response_array_k_t, response_array_k_t))
            performance_k_t = twocat_sextuple_1d(y_valid['fitness'], response_array_k_t)

            print("Test: RBF: Accuracy {}, Precision {}, Recall {}, F1-score {}, ROC AUC {}, PRC AUC {}".format(*performance_k_t))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_t], v_fname)

    # # Not list models .. etc. 
    # mdl.fit(dist_train.values, dist_y_train)
    # dist_array = mdl.predict(dist_test.values).T

    # if rbf_gamma is not None:
    #     response_array = reconstruct.rbf(dist_array, y_train["fitness"], anchors_idx, rbf_gamma, False)
    # else:
    #     response_array = reconstruct.knn(dist_array, y_train["fitness"], anchors_idx, knn=knn)

    # performance = sextuple(y_test["fitness"], response_array.ravel(), False)

    # print("Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}".format(*performance))
    # if rbf_gamma is not None:
    #     write_perform(["VHSE", mdl, n_anchors, "rbf {}".format(rbf_gamma), *performance])
    # else:
    #     write_perform(["VHSE", mdl, n_anchors, "{}-NN".format(knn), *performance])
#%%
RANDOM_SEED = 20
SAMPLE_FRAC = 0.01

# run VHSE topo:
# VHSE_topo(500, [LR()], [1,2,3,4,5], [1,3,5,10])
# raise




# _, df = loaddata.load_Sci_embeddings()
# _, df = loaddata.load_Bert_embeddings('avg')
# 其实顺序不一样。。
# _ = pd.read_parquet("../Datasets/Science21_virus_seq/from_HPC_VHSE_embedding.parquet", engine='fastparquet')
# df = pd.read_parquet("./temp/fullset_df.parquet", engine='fastparquet')
_, df = loaddata.load_VHSE()
# _, df = loaddata.load_VHSE_Expression()
# df = df.iloc[:100]
assert (_.index == df.index).all()  # 证明MPI 是阻塞的结果，顺序是不变的。

df = convert_to_clf_data(df)

# ---- For testing ----
df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
_ = _.reindex(df.index)

_ = _.values.reshape(len(df), -1, 8)


X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

# for mdl in [RFR(n_jobs=-1, verbose=True, min_samples_leaf=3),
#             Ridge(), LR()]:
#     mdl.fit(X_train.reshape(len(X_train), -1), y_train['fitness'])
#     pred = mdl.predict(X_test.reshape(len(X_test), -1))
#     ref_pref = sextuple(y_test['fitness'].values.ravel(), pred.ravel(), False)
#     write_perform(["VHSE,{},{},{},SEED{}".format(SAMPLE_FRAC, str(mdl).replace(",", ";"),
#                                                  len(_), RANDOM_SEED), *ref_pref])

del df
# raise
#%%

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

    layers.Dense(2, activation="softmax")
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9),  # Optimizer
    # optimizer=keras.optimizers.SGD(lr=5e-4, momentum=0.5),
    # Loss function to minimize
    # loss=keras.losses.MeanSquaredError(),
    loss="categorical_crossentropy",
    # List of metrics to monitor
    metrics=["AUC"]
    # metrics=["accuracy"]
)
#%%
print("Fit model on training data")
y_tr = np.hstack((1 - y_train['fitness'].values.reshape(-1, 1), 
            y_train['fitness'].values.reshape(-1, 1))).astype('float32')
y_val = np.hstack((1 - y_valid['fitness'].values.reshape(-1, 1), 
            y_valid['fitness'].values.reshape(-1, 1))).astype('float32')

history = model.fit(
    X_train,
    y_tr,
    batch_size=64,
    # epochs=200,
    epochs=10,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_valid, y_val),
)


np.save("./Outputs/CoVClf/cnn_valid_results.npy", model.predict(X_valid))
predictions = model.predict(X_test)
np.save("./Outputs/CoVClf/cnn_results.npy", predictions)
print("predictions shape:", predictions.shape)
performance = twocat_sextuple_1d(y_test['fitness'].values, predictions)
print(performance)
valid_err = history.history['val_loss'][-1]

write_perform(["Keras VHSE CNN,{},{},SEED{}".format(SAMPLE_FRAC, len(_), RANDOM_SEED), valid_err, *performance])

def myprint(s):
    with open('./Outputs/CoVClf/Results_CoV_VHSE_CNN.txt','a') as f:
        print(s, file=f)

model.summary(print_fn=myprint)

#%%
del X_train, X_valid, X_test
# intermediate_layer_model = keras.Model(inputs=model.input,
#                                  outputs=model.get_layer('max_pooling1d_1').output)

# # embed = pd.read_parquet("../Datasets/Science21_virus_seq/from_HPC_VHSE_embedding.parquet", engine='fastparquet')
# embed = pd.read_parquet("data/fitness_embeddings/VHSE_embedding_matched.parquet", engine='fastparquet')
# embed = embed.values.reshape(len(embed), -1, 8)

# intermediate_output = intermediate_layer_model.predict(embed)
# np.save("./temp/cnn_embedding_orderedsameasmatched.npy", intermediate_output)  # 顺序是不变的。

# embed = pd.read_parquet("data/fitness_embeddings/from_HPC_VHSE_embedding.parquet", engine='fastparquet')
# embed = embed.values.reshape(len(embed), -1, 8)

# intermediate_output = intermediate_layer_model.predict(embed)
# np.save("./temp/cnn_embedding_orderedsameasfromhpc.npy", intermediate_output)  # 顺序是不变的。
