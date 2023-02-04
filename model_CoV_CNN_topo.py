# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:30:10 2021

@author: RUIBZHAN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import loaddata
from topo_reg import calculate_distance, reconstruct
from myToolbox.Metrics import octuple
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics.pairwise import pairwise_distances

def write_perform(mssg, file_name="Results_CoV_VHSE_CNN.csv"):
    string = "\n{}" + ",{}"*(len(mssg)-1)
    with open("./Outputs/" + file_name, 'a') as file:
        file.write(string.format(*mssg))

#%%
RANDOM_SEED = 20
n_anchors = 500
# 11/19/21 run 三个Frac。距离都是Sum of Euclidean along seqeunce residuals
FRAC = 1
output_file = "CNN-TOPO"

df_raw = pd.read_parquet("./temp/fullset_df.parquet")
nat_seq = ["I{}".format(x) for x in range(len(df_raw))]
actual_ii = df_raw.index.get_indexer(nat_seq)
df_raw = df_raw.reindex(nat_seq)
# 11/19 加 100% 的情况
if FRAC == 1:
    embed = np.load("./temp/cnn_embedding_orderedsameasfromhpc.npy")
else:
    embed = np.load("./temp/cnn_embedding_trainedwith{}.npy".format(FRAC))

embed = embed[actual_ii]
# Note, the 3 embeddings are all generated at seed 20. Should keep the seed consistant. 

# Note: 20 units along the seq. 128 channels after CNN. 
#%%
df = df_raw.sample(frac=FRAC, random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

mssg = ["N samples:", len(df), "n training:", len(X_train),
       "n valid:", len(X_valid), "n test:", len(X_test),
       "Data Fraction:", FRAC, "Random Seed:", 20]
write_perform(mssg, "Results_{}_validset.csv".format(output_file))
write_perform(mssg, "Results_{}_testset.csv".format(output_file))

n_anchors = min(len(y_train), n_anchors)
anchors_idx = calculate_distance.select_anchor_idx(y_train, y_train, n_anchors)
ii_df = df_raw.index.get_indexer(df.index)
ii_anchors_embed = df_raw.index.get_indexer(anchors_idx)
ii_anchors_train = y_train.index.get_indexer(anchors_idx)

dist_all = np.zeros((len(df), len(anchors_idx)), dtype=np.float32)
for ii in range(embed.shape[1]):
    print(ii)
    dist_all += pairwise_distances(embed[ii_df, ii, :], embed[ii_anchors_embed, ii, :])
dist_all = pd.DataFrame(dist_all, index=df.index, columns=anchors_idx)

dist_train = dist_all.loc[y_train.index]
dist_valid = dist_all.loc[y_valid.index]
dist_test = dist_all.loc[y_test.index]

dist_y_train = pairwise_distances(y_train["fitness"].values.reshape(-1, 1),
                                  y_train["fitness"].values.reshape(-1, 1)[ii_anchors_train, :])
#%%

mdl = [LR()]
rbf_gamma = [0.1, 1, 2]
knn = [1, 5]
submtr = 'cnn embedding'
metric = 'Sum Euclid along seq length'
for each_mdl in mdl:
    print(each_mdl)
    each_mdl.fit(dist_train.values, dist_y_train)
    dist_array_valid = each_mdl.predict(dist_valid.values).T
    dist_array_test = each_mdl.predict(dist_test.values).T

    for each_gamma in rbf_gamma:
        response_array_r_v = reconstruct.rbf(dist_array_valid, y_train["fitness"], anchors_idx, each_gamma, False)
        performance_r_v = octuple(y_valid['fitness'], response_array_r_v.ravel(), False)[:6]
        print("Valid: RBF: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_r_v))
        write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                       n_anchors, "rbf {}".format(each_gamma),
                       *performance_r_v], "Results_{}_validset.csv".format(output_file))

        response_array_r_t = reconstruct.rbf(dist_array_test, y_train["fitness"], anchors_idx, each_gamma, False)
        performance_r_t = octuple(y_test['fitness'], response_array_r_t.ravel(), False)[:6]
        print("Test: RBF: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_r_t))
        write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                       n_anchors, "rbf {}".format(each_gamma),
                       *performance_r_t], "Results_{}_testset.csv".format(output_file))

    for each_k in knn:
        response_array_k_v = reconstruct.knn(dist_array_valid, y_train["fitness"], anchors_idx, knn=each_k)
        performance_k_v = octuple(y_valid['fitness'], response_array_k_v.ravel(), False)[:6]
        print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_v))
        write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                       n_anchors, "{}-NN".format(each_k),
                       *performance_k_v], "Results_{}_validset.csv".format(output_file))

        response_array_k_t = reconstruct.knn(dist_array_test, y_train["fitness"], anchors_idx, knn=each_k)
        performance_k_t = octuple(y_test['fitness'], response_array_k_t.ravel(), False)[:6]
        print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_t))
        write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                       n_anchors, "{}-NN".format(each_k),
                       *performance_k_t], "Results_{}_testset.csv".format(output_file))
        
#%%
