# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:02:03 2020

@author: ruibzhan
"""

import numpy as np
import pandas as pd
import pickle

from scipy.stats import pearsonr

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor as RFR

# import loaddata, calculate_distance, model_NCI
#%%
def _inv(x):
    return 1 / x

def knn(dist_array, response_train, anchors_idx, knn=10):
    """
    For each test sample, find its nearest k anchors.
    Find the weighted sum of the nearest anchor responses.

    Distance is first transformed to weights,
    and normalized by sum of weights for each sample.
    The sum of weights to all anchors (from a test point) is 1.


    :param dist_array: predicted pair-wise distance. n_anchor x n_test
    :type dist_array: np.array
    :param response_train: all the responeses data.
    :type response_train: pd.DataFrame
    :param anchors_idx: used to select the anchor responses.
    :type anchors_idx: pd.index
    :param knn: How many anchors are contributed to the prediction. defaults to 10
    :type knn: int
    :return: the prediction response array, n_test x n_dim
    :rtype: np.array

    """
    #  Cut off duistance at 0, added 1/16:
    dist_array = np.clip(dist_array, 1e-3, None)

    response_prediction = []
    response_real_values = response_train.loc[anchors_idx]
    if response_real_values.ndim == 1:
        response_real_values = response_real_values.values.reshape(-1, 1)
    for jj in range(dist_array.shape[1]):
        w = dist_array[:, jj]
        assert(len(w) == len(anchors_idx))
        w_inv = w ** -1
        w_inv *= (np.argsort(np.argsort(w_inv)) >= len(w_inv) - knn)  # k-nearest neighbors
        each_response = (response_real_values * w_inv.reshape(-1, 1)).sum(axis=0) / sum(w_inv)
        # response_real_values is n_anchors x n_cell lines. numpy * 此处是每一列乘以w_inv. Uni-var 需要 reshape(-1, 1)
        response_prediction.append(each_response)
    response_array = np.vstack(response_prediction)
    return response_array

def knn2(dist_array, response_train, anchors_idx, knn=10):
    """
    Vectorized knn. Faster.
    The difference is, normalize the weight matrix first,
    then find the nearest neighbors.
    Not reasonable actually. Some points will be too small, since the weights are averaged.
    But lets see if there're any difference.

    :param dist_array: predicted pair-wise distance. n_anchor x n_test
    :type dist_array: np.array
    :param response_train: all the responeses data.
    :type response_train: pd.DataFrame
    :param anchors_idx: used to select the anchor responses.
    :type anchors_idx: pd.index
    :param knn: How many anchors are contributed to the prediction. defaults to 10
    :type knn: int
    :return: the prediction response array, n_test x n_dim
    :rtype: np.array

    """
    n_a, n_t = dist_array.shape
    response_real_values = response_train.loc[anchors_idx]
    if response_real_values.ndim == 1:
        response_real_values = response_real_values.values.reshape(-1, 1)
    inv_v = np.vectorize(_inv)
    w_raw = inv_v(dist_array)
    w_sum = w_raw.sum(axis=0)
    w_norm = w_raw @ np.diag(1/w_sum)
    assert np.allclose(w_norm.sum(axis=0), 1)
    for ii in range(n_t):
        w_norm[:, ii] *= (np.argsort(np.argsort(w_norm[:, ii])) >= (n_a - knn))
        # Note: here is to select the highest weights, not the min distance.
    rt = w_norm.T @ response_real_values.values
    return rt

def test_knn():
    padel_dist="euclidean"
    resp_dist="euclidean"
    kf = KFold(shuffle=True, random_state=2020)
    padel, response = loaddata.load_NCI_responses_padel()
    padel = padel.iloc[:5000]
    response = response.iloc[:5000]
    for train_index, test_index in kf.split(padel):
        pass
    padel_train = padel.iloc[train_index]
    response_train = response.iloc[train_index]
    padel_test = padel.iloc[test_index]
    response_test = response.iloc[test_index]
    anchors_idx = calculate_distance.select_anchor_idx(padel_train, response_train, 200)

    dist_padel_train = calculate_distance.simple_x_train(padel_train, anchors_idx, padel_dist)
    dist_response_train = calculate_distance.simple_y_train(response_train, anchors_idx, resp_dist)
    dist_padel_test = calculate_distance.simple_x_test(padel_train, padel_test, anchors_idx, padel_dist)

    dist_prediction = []

    for ii in range(len(anchors_idx)):
        print("\r{} / {}".format(ii+1, len(anchors_idx)), end='')
        mdl = LR()
        mdl.fit(dist_padel_train, dist_response_train[ii, :])
        single_dist_prediction = mdl.predict(dist_padel_test)
        dist_prediction.append(single_dist_prediction)
    dist_array = np.vstack(dist_prediction)  # m x n_test

    response_prediction = []
    response_real_values = response_train.loc[anchors_idx]

    for jj in range(len(test_index)):
        w = dist_array[:, jj]
        assert(len(w) == len(anchors_idx))
        w_inv = w ** -1
        w_inv *= (np.argsort(np.argsort(w_inv)) >= len(w_inv) - 10)  # k-nearest neighbors
        each_response = (response_real_values * w_inv.reshape(-1, 1)).sum(axis=0) / sum(w_inv)
        response_prediction.append(each_response)

    response_array_1 = np.vstack(response_prediction)
    t1 = time.time()
    response_array_2 = knn(dist_array, response_train, anchors_idx, 10)
    t2 = time.time()
    response_array_3 = knn2(dist_array, response_train, anchors_idx, 10)
    t3 = time.time()
    print("\nknn1: {}\nknn2: {}".format(t2 - t1, t3 - t2))
    assert(np.allclose(response_array_1, response_array_2))

def _rbf(x, s):
    return np.exp(-(x/s)**2)

def rbf(dist_array, response_train, anchors_idx, gamma=1, debug_plot=False) -> np.array:
    """

    :param dist_array: distance array predicted. n_anchors x n_test
    :type dist_array: np.array
    :param response_train: DESCRIPTION
    :type response_train: pd.DataFrame
    :param anchors_idx: DESCRIPTION
    :type anchors_idx: TYPE
    :param gamma: If None, defaults to 1.0.
    :type gamma: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    #  Cut off duistance at 0, added 1/16:
    dist_array = np.clip(dist_array, 1e-3, None)

    n_a, n_t = dist_array.shape
    response_real_values = response_train.loc[anchors_idx]
    if response_real_values.ndim == 1:
        response_real_values = response_real_values.values.reshape(-1, 1)
    if gamma is None:
        # gamma = 1 / response_real_values.shape[1]
        # gamma = np.mean(response_real_values.values.ravel(), axis=None)
        gamma = np.mean(dist_array, axis=None)

    rbf_v = np.vectorize(_rbf)
    k = rbf_v(dist_array, gamma).T  # rbf of distance. n_t x n_a
    h = np.linalg.inv(np.diag(k.sum(axis=1)))  # normalize mat. n_test x n_test
    r = np.asarray(response_real_values)# .values  # real y. n_anchors x n_features.
    rt = h @ k @ r  # np.matmul. Does it work?
    if debug_plot:
        t = h@k
        import seaborn as sns
        sns.distplot(t[2, :], bins=50)
    return rt

def test_rbf():
    padel_dist="euclidean"
    resp_dist="euclidean"
    kf = KFold(shuffle=True, random_state=2020)
    padel, response = loaddata.load_NCI_responses_padel()
    padel = padel.iloc[:125]
    response = response.iloc[:125]
    for train_index, test_index in kf.split(padel):
        pass
    padel_train = padel.iloc[train_index]
    response_train = response.iloc[train_index]
    padel_test = padel.iloc[test_index]
    response_test = response.iloc[test_index]
    anchors_idx = calculate_distance.select_anchor_idx(padel_train, response_train, 20)

    dist_padel_train = calculate_distance.simple_x_train(padel_train, anchors_idx, padel_dist)
    dist_response_train = calculate_distance.simple_y_train(response_train, anchors_idx, resp_dist)
    dist_padel_test = calculate_distance.simple_x_test(padel_train, padel_test, anchors_idx, padel_dist)

    dist_prediction = []

    for ii in range(len(anchors_idx)):
        print("\r{} / {}".format(ii+1, len(anchors_idx)), end='')
        mdl = LR()
        mdl.fit(dist_padel_train, dist_response_train[ii, :])
        single_dist_prediction = mdl.predict(dist_padel_test)
        dist_prediction.append(single_dist_prediction)
    dist_array = np.vstack(dist_prediction)  # m x n_test

    response_array_1 = rbf(dist_array, response_train, anchors_idx, 0.5, True)
    # gamma = 0.5, 选了1个nn， 1000点也均匀选出4个点； gamma = 1， 选了3-4 个；gamma = 2， nn 有10 个，次nn 有十几个。选参可以在0.5 - 2 之间试试。
    response_array_2 = knn(dist_array, response_train, anchors_idx, 10)

    response_prediction = []
    response_real_values = response_train.loc[anchors_idx]
    for jj in range(len(test_index)):
        w = dist_array[:, jj]
        assert(len(w) == len(anchors_idx))
        w_inv = _rbf(w, 5)
        each_response = (response_real_values * w_inv.reshape(-1, 1)).sum(axis=0) / sum(w_inv)
        response_prediction.append(each_response)
    response_array_3 = np.vstack(response_prediction)

#%%
if __name__ == "__main__":
    import time
    # test_knn()
    test_rbf()

