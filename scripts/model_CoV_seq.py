# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:00:55 2021

@author: Ruibo

Run in topo_reg/
"""
import numpy as np
import pandas as pd
import pickle

from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.metrics.pairwise import pairwise_distances
import loaddata, calculate_distance, reconstruct
from myToolbox.Metrics import corr_and_error, sextuple

#%%
def write_perform(mssg, file_name="Results_CoV_intersected.csv"):
    string = "\n{}" + ",{}"*(len(mssg)-1)
    with open("./Outputs/" + file_name, 'a') as file:
        file.write(string.format(*mssg))

#%%
def test_precompute_knn():
    """

    """
    kf = KFold(shuffle=True, random_state=2020)
    from sklearn.datasets import load_boston
    d = load_boston()
    x = d['data']
    y = d['target']
    dist_all = pd.DataFrame(pairwise_distances(x))
    for train_index, test_index in kf.split(dist_all):
        x_train = dist_all.iloc[train_index, train_index]
        y_train = y[train_index]
        x_test = dist_all.iloc[test_index, train_index]
        y_test = y[test_index]
        mdl = KNN(metric='precomputed').fit(x_train, y_train)
        pred = mdl.predict(x_test)
        performance = sextuple(y_test.ravel(), pred.ravel(), False)

        print("Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}\nNMAE: {}".format(*performance))

        mdl0 = KNN().fit(x[train_index], y[train_index])
        perform0 = sextuple(y[test_index], mdl0.predict(x[test_index]), False)
        print("Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}\nNMAE: {}".format(*perform0))
        print("=====================================")


def ref_model():
    kf = KFold(shuffle=True, random_state=2020)
    data = loaddata.load_CoV_fitness()
    # ---- For testing ----
    data = data.loc[SAMPLES]
    dist_all = pd.DataFrame(
        calculate_distance.Levenshtein_x_train_p(data, None, data.index),
        index=data.index)
    for train_index, test_index in kf.split(data):
        x_train = dist_all.iloc[train_index, train_index]
        y_train = data.iloc[train_index, 1]
        x_test = dist_all.iloc[test_index, train_index]
        y_test = data.iloc[test_index, 1]
        mdl = KNN(metric='precomputed').fit(x_train, y_train)
        pred = mdl.predict(x_test)
        performance = sextuple(y_test.values.ravel(), pred.ravel(), False)

        print("Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}".format(*performance))
        write_perform(["Reference KNN,,,", *performance])

#%%
def SW_distance_holdout(n_anchors=100, mdl=LR(), metric='Levenshtein', knn=10,
                        geodesic=False, rbf_gamma=None, submtr=None, dset='fitness', **kwargs):

    # ---- Try out a bunch of rbfs and knns at the same time ----

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

    # ---- Load data ----
    # _, df = loaddata.load_Sci_embeddings()
    _, df = loaddata.load_Sci_embeddings(dataset=dset)
    redo_index = ["I{}".format(x) for x in range(len(df))]
    _.index = redo_index
    df.index = redo_index

    # df = pd.read_parquet("./temp/fullset_df.parquet", engine='fastparquet')  #temporary match to cnn results
    
    # ---- For testing ----
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    _ = _.reindex(df.index)

    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

    if dset == 'fitness':
        v_fname = 'Results_heldout_validset.csv'
        t_fname = "Results_heldout_testset.csv"
    elif dset == 'expression':
        v_fname = 'ResultsExp_heldout_validset.csv'
        t_fname = "ResultsExp_heldout_testset.csv"
    else:
        raise ValueError

    mssg = ["N samples:", len(df), "n training:", len(X_train),
           "n valid:", len(X_valid), "n test:", len(X_test),
           "Data Fraction:", SAMPLE_FRAC, "Random Seed:", RANDOM_SEED]
    write_perform(mssg, v_fname)
    write_perform(mssg, t_fname)

    if n_anchors is None:
        n_anchors = len(X_train)
    anchors_idx = calculate_distance.select_anchor_idx(y_train, y_train, n_anchors)
    """
    Logic: 后面计算大样本量，n_anchors << n_samples. anchors 每次基本是全新的，不能重复利用，没必要在loop 外计算
    """
    if "leven" in metric.lower():
        dist_all = pd.DataFrame(
            calculate_distance.Levenshtein_x_train_p(df, None, anchors_idx),
            index=df.index)
    elif "sw" or "waterman" in metric.lower():
        dist_all = pd.DataFrame(
            calculate_distance.SW_x_train_p(df, None, anchors_idx, submtr=submtr, **kwargs),
            index=df.index)
    elif "ct" or "conjoint" in metric.lower():
        dist_all = pd.DataFrame(
            calculate_distance.CT_x_train(df, None, anchors_idx, **kwargs),
            index=df.index)
    dist_all.columns = anchors_idx
    dist_train = dist_all.loc[y_train.index]
    dist_valid = dist_all.loc[y_valid.index]
    dist_test = dist_all.loc[y_test.index]

    ii_anchors = y_train.index.get_indexer(anchors_idx)
    dist_y_train = pairwise_distances(y_train["fitness"].values.reshape(-1, 1))[:, ii_anchors]

    #%% modelling of distances
    for each_mdl in mdl:
        each_mdl.fit(dist_train.values, dist_y_train)
        dist_array_valid = each_mdl.predict(dist_valid.values).T
        dist_array_test = each_mdl.predict(dist_test.values).T
        # 对 LR 正反距离都一样
        for each_gamma in rbf_gamma:
            response_array_r_v = reconstruct.rbf(dist_array_valid, y_train["fitness"], anchors_idx, each_gamma, False)
            performance_r_v = sextuple(y_valid['fitness'], response_array_r_v.ravel(), False)
            print("Valid: RBF: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_r_v))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "rbf {}".format(each_gamma),
                           *performance_r_v], v_fname)
            np.save("topo_valid_result.npy", response_array_r_v)  #TODO: temporary
            np.save("actual_valid.npy", y_valid['fitness'].values)
            
            response_array_r_t = reconstruct.rbf(dist_array_test, y_train["fitness"], anchors_idx, each_gamma, False)
            performance_r_t = sextuple(y_test['fitness'], response_array_r_t.ravel(), False)
            print("Test: RBF: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_r_t))
            write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "rbf {}".format(each_gamma),
                           *performance_r_t], t_fname)
            np.save("topo_result.npy", response_array_r_t)  #TODO: temporary
            np.save("actual_result.npy", y_test['fitness'].values)

        for each_k in knn:
            response_array_k_v = reconstruct.knn(dist_array_valid, y_train["fitness"], anchors_idx, knn=each_k)
            performance_k_v = sextuple(y_valid['fitness'], response_array_k_v.ravel(), False)
            print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_v))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_v], v_fname)

            response_array_k_t = reconstruct.knn(dist_array_test, y_train["fitness"], anchors_idx, knn=each_k)
            performance_k_t = sextuple(y_test['fitness'], response_array_k_t.ravel(), False)
            print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_t))
            write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_t], t_fname)


def SW_distance_fullset(n_anchors=200, mdl=LR(), knn=10, rbf_gamma=None):

    # ---- Try out a bunch of rbfs and knns at the same time ----

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

    # ---- Load data ----
    _, df = loaddata.load_Sci_embeddings()  # for binding dataset
    # _, df = loaddata.load_Sci_embeddings(dataset='expression')  # for expression dataset
    
    redo_index = ["I{}".format(x) for x in range(len(df))]  # avoid confusion and error in parquet
    _.index = redo_index
    df.index = redo_index

    # ---- For testing ----
    df = df.sample(frac=1, random_state=20)
    _ = _.reindex(df.index)

    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=20)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=20)

    mssg = ["N samples:", len(df), "n training:", len(X_train),
           "n valid:", len(X_valid), "n test:", len(X_test),
           "Data Fraction:", 1, "Random Seed:", 20]
    write_perform(mssg, "Results_fullset_validset.csv")
    write_perform(mssg, "Results_fullset_testset.csv")

    anchors_idx = calculate_distance.select_anchor_idx(y_train, y_train, n_anchors)
    # anchors_idx.to_csv("Fullset_test_selected_{}anchors.csv")

    dist_all = pd.read_parquet("./temp/反距离/from_HPC_{}anchors_Seed20.parquet".format(n_anchors))  # for binding dataset
    # dist_all = pd.read_parquet("./temp/from_HPC_WholeSet500anchors_expression_Seed20.parquet")  # for expression dataset
    
    dist_all = dist_all.max() - dist_all  # TODO: temp   # for binding dataset
    dist_train = dist_all.loc[y_train.index]
    dist_valid = dist_all.loc[y_valid.index]
    dist_test = dist_all.loc[y_test.index]
    print("dist loaded.. ")
    ii_anchors = y_train.index.get_indexer(anchors_idx)
    dist_y_train = pairwise_distances(y_train["fitness"].values.reshape(-1, 1),
                                      y_train["fitness"].values.reshape(-1, 1)[ii_anchors, :])
    # if geodesic:
    #     dist_response_train = calculate_distance.isomap_L(response_train, anchors_idx, n_neighbors=5)

    #%% modelling of distances
    submtr = 'blosum62' #TODO   # or BLOSUM50? 11/19 reviewing
    metric = 'SW'
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
                           *performance_r_v], "Results_full_testset.csv")
            np.save("topo_valid_result.npy", response_array_r_v)  #TODO: temporary
            np.save("actual_valid.npy", y_valid['fitness'].values)
            response_array_r_t = reconstruct.rbf(dist_array_test, y_train["fitness"], anchors_idx, each_gamma, False)
            performance_r_t = sextuple(y_test['fitness'], response_array_r_t.ravel(), False)
            print("Test: RBF: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_r_t))
            write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "rbf {}".format(each_gamma),
                           *performance_r_t], "Results_full_testset.csv")
            np.save("topo_result.npy", response_array_r_t)  #TODO: temporary
            np.save("actual_result.npy", y_test['fitness'].values)
            
        for each_k in knn:
            response_array_k_v = reconstruct.knn(dist_array_valid, y_train["fitness"], anchors_idx, knn=each_k)
            performance_k_v = sextuple(y_valid['fitness'], response_array_k_v.ravel(), False)
            print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_v))
            write_perform(["Validation", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_v], "Results_full_testset.csv")
            np.save("topo_valid_result.npy", response_array_k_v)  #TODO: temporary
            np.save("actual_valid.npy", y_valid['fitness'].values)
            
            response_array_k_t = reconstruct.knn(dist_array_test, y_train["fitness"], anchors_idx, knn=each_k)
            performance_k_t = sextuple(y_test['fitness'], response_array_k_t.ravel(), False)
            print("knn: Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_k_t))
            write_perform(["Testing", metric, submtr, str(each_mdl).replace(",", ";"),
                           n_anchors, "{}-NN".format(each_k),
                           *performance_k_t], "Results_full_testset.csv")
            np.save("topo_result.npy", response_array_k_t)  #TODO: temporary
            np.save("actual_result.npy", y_test['fitness'].values)

#%%
def embedding_ref_mdl(ref_mdl = LR(), dataset='ProtBert', pooling='avg'):
    """
    bert or sci
    """
    output_file = "Results_fitness_embedding_RF"
    if isinstance(ref_mdl, BaseEstimator):
        ref_mdl = [ref_mdl]
    try:
        if dataset == 'sci':
            _, df = loaddata.load_Sci_embeddings(pooling, dataset='fitness')
            # _, df = loaddata.load_Sci_embeddings(pooling, dataset='expression')
        elif 'bert' in dataset.lower():
            _, df = loaddata.load_Bert_embeddings(pooling, dataset='fitness')
            # _, df = loaddata.load_Bert_embeddings(pooling, dataset='expression')
    except ValueError:
        return None

    redo_index = ["I{}".format(x) for x in range(len(df))]
    _.index = redo_index
    df.index = redo_index

    # ---- For testing ----
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    _ = _.reindex(df.index)

    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

    for each_mdl in ref_mdl:
        each_mdl.fit(X_train, y_train["fitness"])
        pred_v = each_mdl.predict(X_valid)
        pred_t = each_mdl.predict(X_test)
        performance_v = sextuple(y_valid['fitness'], pred_v, False)   #TODO
        performance_t = sextuple(y_test['fitness'], pred_t, False)
        print("Test\nSpearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance_t))
        write_perform(["Validation,Ref mdl", dataset, "{} Pooling".format(pooling),
                       str(each_mdl).replace(",", ";"), *performance_v],
                      "{}_validset.csv".format(output_file))
        write_perform(["Test,Ref mdl", dataset, "{} Pooling".format(pooling),
                       str(each_mdl).replace(",", ";"), *performance_t],
                      "{}_testset.csv".format(output_file))
    return None


def knn_ref_holdout_model():

    _, df = loaddata.load_Sci_embeddings()
    # print(_.shape)
    redo_index = ["I{}".format(x) for x in range(len(df))]
    _.index = redo_index
    df.index = redo_index

    # ---- For testing ----
    df = df.sample(frac=0.05, random_state=RANDOM_SEED)
    _ = _.reindex(df.index)

    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

    # ---- calculate distance ----
    # data = pd.concat((y_train, y_valid))
    # dist_all = pd.DataFrame(
    #     calculate_distance.SW_x_train_p(data, None, data.index),
    #     index=data.index, columns=data.index)
    # ---- Load distance ----
    dist_all = pd.read_parquet("./temp/from_HPC_allpairs_forKNN_Seed{}.parquet".format(RANDOM_SEED))
    dist_all = dist_all.max() - dist_all
    x_train = dist_all.loc[X_train.index, X_train.index]
    y_train = df.loc[X_train.index, 'fitness']
    x_test = dist_all.loc[X_test.index, X_train.index]
    y_test = df.loc[X_test.index, 'fitness']
    mdl = KNN(metric='precomputed', n_neighbors=20).fit(x_train, y_train)
    pred = mdl.predict(x_test)
    performance = sextuple(y_test.values.ravel(), pred.ravel(), False)

    print("Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance))
    write_perform(["Reference KNN,SW,blosum55,seed20,5-nn", *performance], "Results_heldout_testset.csv")


def knn_ref_small_holdout_model(knns=[1, 3, 5, 10, 20]):
    _, df = loaddata.load_Sci_embeddings()
    # print(_.shape)
    redo_index = ["I{}".format(x) for x in range(len(df))]
    _.index = redo_index
    df.index = redo_index

    # ---- For testing ----
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    _ = _.reindex(df.index)

    X_train, X_test, y_train, y_test = train_test_split(_, df, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

    dist_all = pd.DataFrame(
            calculate_distance.SW_x_train_p(df, None, df.index, submtr="blosum55"),
            index=df.index, columns=df.index)

    # dist_all = dist_all.max() - dist_all

    x_train = dist_all.loc[X_train.index, X_train.index]
    y_train = df.loc[X_train.index, 'fitness']
    x_test = dist_all.loc[X_test.index, X_train.index]
    y_test = df.loc[X_test.index, 'fitness']

    for knn in knns:
        mdl = KNN(metric='precomputed', n_neighbors=knn).fit(x_train, y_train)
        pred = mdl.predict(x_test)
        performance = sextuple(y_test.values.ravel(), pred.ravel(), False)

        print("Spearman: {}\nPearson: {}\nMSE: {}\nMAE: {}\nNRMSE: {}".format(*performance))
        write_perform(["Reference KNN,SW,blosum55,seed{},{}-nn".format(RANDOM_SEED, knn),
                        *performance], "Results_heldout_testset.csv")

#%%
if __name__ == "__main__":

    # sample size selection
    # SAMPLE_FRAC = 0.001  # for test run
    # SAMPLE_FRAC = 0.005
    SAMPLE_FRAC = 0.05
    # SAMPLE_FRAC = 1
    RANDOM_SEED = 20


    # 11/29/2021  Topo regression
    mdls = [RFR(n_jobs=-1, verbose=True, min_samples_leaf=3),
            Ridge(),
            LR()]
    SW_distance_holdout(100, metric='sw', mdl=mdls, rbf_gamma=[0.5, 1, 5, 10], knn=[1, 3, 5, 10])
    SW_distance_holdout(500, metric='sw', mdl=mdls, rbf_gamma=[0.5, 1, 5, 10], knn=[1, 3, 5, 10])
    # Full set when 100% of dataset is used.
    SW_distance_fullset(200, mdl=mdls, rbf_gamma=[0.5, 1, 2, 5, 10], knn=[1, 3, 5, 10])
    SW_distance_fullset(500, mdl=mdls, rbf_gamma=[0.5, 1, 2, 5, 10], knn=[1, 3, 5, 10])


    # reference model KNN
    knn_holdout_ref_model()

    
    # For LSTM or BERT embedding -> Common regression model
    # for pl in ['avg', 'last', 'cscs', 'cls']:
    for pl in ['avg', 'last', 'cls']:
        for ds in ["ProtBert", "sci"]:
            if pl == 'cls' and ds == 'sci':
                continue
            else:
                embedding_ref_mdl(mdls, ds, pl)
    