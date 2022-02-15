import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import loaddata, calculate_distance, reconstruct
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from myToolbox.Metrics import sextuple
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
def VHSE_topo(n_anchors=500, mdl=LR(), rbf_gamma=1, knn=5):
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
    for each_mdl in mdl:
        print(each_mdl)
        each_mdl.fit(dist_train.values, dist_y_train)
        dist_array_valid = each_mdl.predict(dist_valid.values).T
        dist_array_test = each_mdl.predict(dist_test.values).T
    
        for each_gamma in rbf_gamma:
            response_array_r_v = reconstruct.rbf(dist_array_valid, y_train["fitness"], anchors_idx, each_gamma, False)
            performance_r_v = sextuple(y_valid['fitness'], response_array_r_v.ravel(), False)
            print("Valid: RBF: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_r_v))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "rbf {}".format(each_gamma),
                           *performance_r_v], "Results_{}_validset.csv".format("VHSE_TOPO"))
            response_array_r_t = reconstruct.rbf(dist_array_test, y_train["fitness"], anchors_idx, each_gamma, False)
            performance_r_t = sextuple(y_test['fitness'], response_array_r_t.ravel(), False)
            print("Test: RBF: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_r_t))
            write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "rbf {}".format(each_gamma),
                           *performance_r_t], "Results_{}_testset.csv".format("VHSE_TOPO"))
            
        for each_k in knn:
            response_array_k_v = reconstruct.knn(dist_array_valid, y_train["fitness"], anchors_idx, knn=each_k)
            performance_k_v = sextuple(y_valid['fitness'], response_array_k_v.ravel(), False)
            print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_v))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_v], "Results_{}_validset.csv".format("VHSE_TOPO"))
    
            response_array_k_t = reconstruct.knn(dist_array_test, y_train["fitness"], anchors_idx, knn=each_k)
            performance_k_t = sextuple(y_test['fitness'], response_array_k_t.ravel(), False)
            print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_t))
            write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_t], "Results_{}_testset.csv".format("VHSE_TOPO"))
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

if __name__ == "__main__":
    #%%
    RANDOM_SEED = 20
    SAMPLE_FRAC = 1

    VHSE_topo(500, [LR()], [1,2,3,4,5], [1,3,5,10])
    raise
