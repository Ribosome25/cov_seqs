# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 21:36:08 2020

@author: ruibzhan
"""
import os
import random
import warnings
from topo_reg.io import check_input_data_df, load_int_dict_from_json, read_df_list, save_df
from topo_reg.args import DistanceArgs, ImageDistArgs, SW_Args
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
import sklearn.manifold as mnf
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

import difflib
#----------------
# import loaddata
# import image_distance

from myToolbox.Stat import top_percentage_distribution


#--------------
# 需要时再comment 吧。以后可以把calc dist 分成chem 和 bio 两个
from skbio.alignment import StripedSmithWaterman as SW
from topo_reg.substitution_matrix import blosum50
from topo_reg.load_sub_matrix import load_subs_mtrx
#%% Entry

def calc_dist(args: DistanceArgs):
    # the general entry function
    if isinstance(args, SW_Args):
        return None
    if isinstance(args, ImageDistArgs):
        return None

    df = read_df_list(args.df_path)
    index = load_int_dict_from_json(args.index_path)
    if len(index['test_idx']) > 0:
        warnings.warn("Test index will be omitted. Use calc_dist_precopmputed instead if want to compute the test distance at the same time.")
    if args.method == "simple":
        dist = simple_x_dist(df, index['train_idx'], index['anchors_idx'], args.metric)
        dist = dist.astype(np.float32)
        save_df(pd.DataFrame(dist, index=index["train_idx"], columns=index['anchors_idx']), args.save_path)
        return dist

def calc_dist_precomputed(args: DistanceArgs):
    df = read_df_list(args.df_path)
    if args.x_columns is None:
        args.x_columns = df.columns
    df = df[args.x_columns]

    idx_dict = load_int_dict_from_json(args.index_path)
    if args.test_index_path is not None:
        test_idx_dict = load_int_dict_from_json(args.test_index_path)
        idx_dict['test_idx'] = test_idx_dict['test_idx']

    train_idx = list(idx_dict['train_idx'])
    anchors_idx = list(idx_dict['anchors_idx'])
    test_idx = list(idx_dict['test_idx'])
    both_idx = train_idx + test_idx

    if args.method == 'simple':
        result = pd.DataFrame(
            simple_x_dist(df, both_idx, anchors_idx, metric=args.metric),
            index=both_idx, columns=anchors_idx)
        save_df(result.loc[train_idx], args.save_path, 'train')
        save_df(result.loc[test_idx], args.save_path, 'test')

    if args.method.lower() == 'sw':
        # Note: collected are similarity scores. 
        # has to collect then reverse. Have to mod the calculate dist py accordingly. 
        result = pd.DataFrame(
            SW_x_train_p(df, both_idx, anchors_idx, submtr=args.sub_matrix, relative=False),
            index=both_idx, columns=anchors_idx)
        result = result.max() - result
        # 其实也不用分开存。每一步都在调用json index 文件。
        save_df(result.loc[train_idx], args.save_path, 'train')
        save_df(result.loc[test_idx], args.save_path, 'test')


#%%
def select_anchor_idx(padel, response, n_anchors, seed=2020):
    if n_anchors is None:
        n_anchors = len(padel)
    random.seed(seed)
    if (response.index == padel.index).all():
        return random.sample(padel.index.tolist(), n_anchors)
    else:
        raise ValueError("Select anchors: 2 inputs have different indice. ")

def pca_anchor_idx(padel, response, n_anchors, seed=2020):
    assert(n_anchors < response.shape[0])
    random.seed(seed)
    if not (response.index == padel.index).all():
        raise ValueError("Select anchors: 2 inputs have different indice. ")
    pca = PCA(n_components=1)
    pc_response = pd.DataFrame(pca.fit_transform(response), index=response.index)
    bins = np.linspace(pc_response.min(), pc_response.max(), n_anchors).ravel()
    inds = np.digitize(pc_response.values, bins).ravel()
    # digitize: Return the indices of the bins to which each value in input array belongs.
#    assert(len(inds) == n_anchors)
    final_set = set()
    while(len(final_set) < n_anchors):
        each_set = set()
        for nbin in range(1, n_anchors + 1):
            try:
                ii = np.argwhere(inds == nbin)[0][0]
                each_set.add(ii)
                inds[ii] = -1
            except IndexError:
                continue

        if (len(final_set) + len(each_set) ) <= n_anchors:
            final_set |= each_set
        else:
            chaduoshao = n_anchors - len(final_set)
            trans_set = set(random.sample(each_set, chaduoshao))
            final_set |= trans_set
    index = response.iloc[list(final_set)].index
    return index

def test_pca_anchor():
    x = np.random.randn(100, 10)
    n = 90
    dfx = pd.DataFrame(x)
    idx = pca_anchor_idx(dfx, dfx, n)
    return idx

def select_near_samples(padel, response, n_samples=200, percent=25):
    idx = select_anchor_idx(padel, response, 5 * n_samples)
    padel = padel.reindex(idx)
    response = response.reindex(idx)
    p_dist = pairwise_distances(padel)
    r_dist = pairwise_distances(response)
    p_thrs = top_percentage_distribution(p_dist, percent, False)
    r_thrs = top_percentage_distribution(r_dist, percent, False)
    p = p_dist < p_thrs
    r = r_dist < r_thrs
    m = p * r
    x = m.sum(axis=0)
    ii = np.argsort(-x)[:n_samples]
    return np.array(idx)[ii]

def select_near_and_far_samples(padel, response, n_samples=200, percent=25):
    half_percent = percent / 2
    idx = select_anchor_idx(padel, response, 5*n_samples)
    padel = padel.reindex(idx)
    response = response.reindex(idx)
    p_dist = pairwise_distances(padel)
    r_dist = pairwise_distances(response)

    p_thrs_h = top_percentage_distribution(p_dist, half_percent, True)
    r_thrs_h = top_percentage_distribution(r_dist, half_percent, True)
    p_thrs_l = top_percentage_distribution(p_dist, half_percent, False)
    r_thrs_l = top_percentage_distribution(r_dist, half_percent, False)

    ph = p_dist > p_thrs_h
    pl = p_dist < p_thrs_l
    rh = r_dist > r_thrs_h
    rl = r_dist < r_thrs_l

    mh = ph * rh
    ml = pl * rl

    m = (mh + ml).astype(bool)

    x = m.sum(axis=0)
    ii = np.argsort(-x)[:n_samples]
    return np.array(idx)[ii]

def select_different(padel, response, n_samples=200, percent=25):
    half_percent = percent / 2
    idx = select_anchor_idx(padel, response, 5*n_samples)
    padel = padel.reindex(idx)
    response = response.reindex(idx)
    p_dist = pairwise_distances(padel)
    r_dist = pairwise_distances(response)

    p_thrs_h = top_percentage_distribution(p_dist, half_percent, True)
    r_thrs_h = top_percentage_distribution(r_dist, half_percent, True)
    p_thrs_l = top_percentage_distribution(p_dist, half_percent, False)
    r_thrs_l = top_percentage_distribution(r_dist, half_percent, False)

    ph = p_dist > p_thrs_h
    pl = p_dist < p_thrs_l
    rh = r_dist > r_thrs_h
    rl = r_dist < r_thrs_l

    ma = ph * rl
    mb = pl * rh

    m = (ma + mb).astype(bool)

    x = m.sum(axis=0)
    ii = np.argsort(-x)[:n_samples]
    return np.array(idx)[ii]

#%%
def simple_x_dist(df, train_idx, anchors_idx, metric):
    # added 9/8/21
    if train_idx is None:
        train_idx = df.index
    if anchors_idx is None:
        anchors_idx = df.index
    train_df = df.loc[train_idx]
    anchors_df = df.loc[anchors_idx]
    dist = pairwise_distances(train_df, anchors_df, metric=metric)
    return dist

def simple_x_train(padel, anchors_idx, metric):
    """
    From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].
    These metrics support sparse matrix inputs.
    [‘nan_euclidean’] but it does not yet support sparse matrices.

    From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’,
     ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’,
     ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    See the documentation for scipy.spatial.distance for details on these metrics.
    These metrics do not support sparse matrix inputs.
    """
    anchors_padel = padel.loc[anchors_idx]
    padel_dist = pairwise_distances(padel, anchors_padel, metric=metric)
    return padel_dist

# def simple_y_train(response, anchors_idx, metric):
def simple_y_train(response, anchors_idx, metric, train_idx=None):
    # A quick fix, considering the compatability. 
    anchors_response = response.loc[anchors_idx]
    if train_idx is not None:
        response = response.loc[train_idx]
    if response.ndim == 1:
        response = response.values.reshape(-1, 1)
        anchors_response = anchors_response.values.reshape(-1, 1)
    response_dist = pairwise_distances(anchors_response, response, metric=metric)
    return response_dist

def simple_x_test(padel, test_padel, anchors_idx, metric):
    anchors_padel = padel.loc[anchors_idx]
    padel_dist = pairwise_distances(test_padel, anchors_padel, metric=metric)
    return padel_dist

def simple_y_test(response_test, response_train, anchors_idx, metric):
    anchors_response = response_train.loc[anchors_idx]
    if response_train.ndim == 1:
        response_test = response_test.values.reshape(-1, 1)
        anchors_response = anchors_response.values.reshape(-1, 1)
    response_dist = pairwise_distances(anchors_response, response_test, metric=metric)
    return response_dist

def test_simple_x_dist():
    arr = np.random.randn(100, 10)
    df = pd.DataFrame(arr)
    anchors = pd.Index(np.random.randint(0, 10, size=5))
    d1 = simple_x_train(df, anchors, 'euclidean')
    d2 = simple_x_dist(df, None, anchors, 'euclidean')
    assert np.allclose(d1, d2)

#%% Isomap geodestic distance
def isomap_x(padel_train, padel_test, anchors_idx, metric, **kwargs):
    ii = padel_train.index.get_indexer_for(anchors_idx)
    iso = mnf.Isomap(metric=metric, path_method='D', **kwargs)
    iso.fit(padel_train)
    d_train = iso.dist_matrix_
    iso.transform(padel_test)
    d_test = iso.testing_dist_matrix_

    # 为什么有全是0的点，对应ref 距离是最远的
    # IsoMap 的距离shortest path fit 出来有一些全是0
    # 用 D 和 FW 是一样

    return d_train[:, ii], d_test[:, ii]

def isomap_y_train(response, anchors_idx, metric, **kwargs):
    ii = response.index.get_indexer_for(anchors_idx)
    iso = mnf.Isomap(metric=metric, **kwargs)
    iso.fit(response)
    d = iso.dist_matrix_
    return d[ii]

def test_isomap_dist():
    padel, response = loaddata.load_NCI_responses_padel()
    SAMPLES = select_anchor_idx(padel, response, 2000)
    padel = padel.loc[SAMPLES]
    response = response.loc[SAMPLES]
    anchors_idx = select_anchor_idx(padel, response, 50)

    d = isomap_y_train(response, anchors_idx, "euclidean", n_neighbors=5)
    dref = simple_y_train(response, anchors_idx, "euclidean")
    t = d + 1e-5 > dref
    assert t.all()

    dx1, dx2 = isomap_x(padel, padel.iloc[1800:], anchors_idx, "euclidean", n_neighbors=10)
    dx1_ref = simple_x_train(padel, anchors_idx, "euclidean")
    dx2_ref = simple_x_test(padel, padel.iloc[1800:], anchors_idx, "euclidean")
    t1 = (dx1 + 1e-5 > dx1_ref)
    t2 = (dx2 + 1e-5 > dx2_ref)
    assert t1.all() & t2.all()
    return d
#%%
def load_manifold_y(file_name, orig_reponse):
    embd_np = np.load(file_name)
    return pd.DataFrame(embd_np, index=orig_reponse.index)

def manifold_y_train(embd_response, anchors_idx, dist_metric="euclidean"):
    anchors_response = embd_response.loc[anchors_idx]
    response_dist = pairwise_distances(anchors_response, embd_response, metric=dist_metric)
    return response_dist

#%%
def pca_x_train(padel, anchors_idx, n_dim, metric):
    pca = PCA(n_components=n_dim)
    pc_padel = pd.DataFrame(pca.fit_transform(padel), index=padel.index)
    pc_anchors_padel = pc_padel.loc[anchors_idx]
    padel_dist = pairwise_distances(pc_padel, pc_anchors_padel, metric=metric)
    return padel_dist, pca

def pca_y_train(response, anchors_idx, n_dim, metric):
    pca = PCA(n_components=n_dim)
    pc_response = pd.DataFrame(pca.fit_transform(response), index=response.index)
    anchors_response = pc_response.loc[anchors_idx]
    response_dist = pairwise_distances(anchors_response, pc_response, metric=metric)
    return response_dist

def pca_x_test(pca, padel, test_padel, anchors_idx, metric):
    assert(isinstance(pca, PCA))
    pc_padel_test = pca.transform(test_padel)
    pc_anchors_padel = pca.transform(padel.loc[anchors_idx])
    return pairwise_distances(pc_padel_test, pc_anchors_padel, metric=metric)
   #%%

def image_x_train(image_path, train_idx, anchors_idx, connectivity, **kwargs):
    """
    Pass the connectivity to select which method, like dilation, or whatever.

    This is a uni-thread for loop version. Basic.

    Accepted connectivites:
        Euclidean, Euclidean binary, Gaussian blurred euclidean,
        Dilation, (undirected) Dilation,
        Grayscale (dilation),
        SSIM,
        Hausdorff, undirected Hausdorff, erosion hausdorff,
        Pooling.

    kwargs:
        k: the size of dilation kernel, or pooling kernel
        sigma: the sigma for Gaussian blurring
        relative: if the distance is normalized by the
        mode: pooling, max or avg.
        stride: pooling window stride. default 1 or 2?
    """
    Nn = len(train_idx)
    Mm = len(anchors_idx)
    dist_mat = np.zeros((Nn, Mm))
    verb = kwargs.get("verbose")
    ii = 1

    for xx in range(Nn):
        idx = train_idx[xx]
        f1 = os.path.join(image_path, "MDS_"+str(idx)+".npy")
        img1 = np.load(f1)
        for yy in range(Mm):
            col = anchors_idx[yy]
            if verb:
                print("\r{} / {}\t".format(ii, Nn*Mm), end='')
                ii += 1
            if idx == col:
                pp_dist = 0
            else:
                f2 = os.path.join(image_path, "MDS_"+str(col)+".npy")
                img2 = np.load(f2)
                pp_dist = image_distance.p2p_distance(img1, img2, connectivity,
                                             **kwargs)
            dist_mat[xx, yy] = pp_dist
    # padel_dist = pairwise_distances(padel, anchors_padel, metric=metric)
    vmax = np.nanmax(dist_mat)
    np.nan_to_num(dist_mat, copy=False, nan=vmax)
    return dist_mat

def _xp_dist(xx, image_path, train_idx, img2, connectivity, **kwargs):
    """
    Element distance calculation func.
    For parallel distance computation.
    """
    idx = train_idx[xx]
    f1 = os.path.join(image_path, "MDS_"+str(idx)+".npy")
    img1 = np.load(f1)
    pp_dist = image_distance.p2p_distance(img1, img2, connectivity,
                                 **kwargs)
    return pp_dist

def image_x_train_p(image_path, train_idx, anchors_idx, connectivity, **kwargs):
    """
    Pass the connectivity to select which method, like dilation, or what ever.
    Multi-thread computing using joblib.

    """
    Nn = len(train_idx)
    Mm = len(anchors_idx)
    dist_mat = np.zeros((Nn, Mm))
    ii = 1

    for yy in range(Mm):
        print("\r{} / {}\t".format(ii, Mm), end='')
        col = anchors_idx[yy]
        f2 = os.path.join(image_path, "MDS_"+str(col)+".npy")
        img2 = np.load(f2)
        pp_dist = Parallel(n_jobs=-1)(delayed(_xp_dist)(xx, image_path, train_idx, img2, connectivity, **kwargs) for xx in range(Nn))
        dist_mat[:, yy] = pp_dist
        ii += 1
    vmax = np.nanmax(dist_mat)
    np.nan_to_num(dist_mat, copy=False, nan=vmax)
    return dist_mat


def image_y_train(image_path, train_idx, anchors_idx, connectivity, **kwargs):
    """
    Uni-thread image distance computing for responses.
    """
    Mm = len(train_idx)
    Nn = len(anchors_idx)
    dist_mat = np.zeros((Nn, Mm))
    verb = kwargs.get("verbose")
    ii = 1

    for xx in range(Nn):
        idx = anchors_idx[xx]
        f1 = os.path.join(image_path, "MDS_"+str(idx)+".npy")
        img1 = np.load(f1)
        for yy in range(Mm):
            col = train_idx[yy]
            if verb:
                print("\r{} / {}\t".format(ii, Nn*Mm), end='')
                ii += 1
            if idx == col:
                pp_dist = 0
            else:
                f2 = os.path.join(image_path, "MDS_"+str(col)+".npy")
                img2 = np.load(f2)
                pp_dist = image_distance.p2p_distance(img1, img2, connectivity,
                                             **kwargs)
            dist_mat[xx, yy] = pp_dist
    # response_dist = pairwise_distances(anchors_reponse, response, metric=metric)
    vmax = np.nanmax(dist_mat)
    np.nan_to_num(dist_mat, copy=False, nan=vmax)
    return dist_mat

def _yp_dist(yy, image_path, train_idx, img1, connectivity, **kwargs):
    col = train_idx[yy]
    f2 = os.path.join(image_path, "MDS_"+str(col)+".npy")
    img2 = np.load(f2)
    pp_dist = image_distance.p2p_distance(img1, img2, connectivity,
                                     **kwargs)
    return pp_dist


def image_y_train_p(image_path, train_idx, anchors_idx, connectivity, **kwargs):
    """
    Parallel response image distance.
    Output: n_anchors x n_training samples.
    """
    Nn = len(anchors_idx)
    Mm = len(train_idx)
    dist_mat = np.zeros((Nn, Mm))
    ii = 1
    for xx in range(Nn):
        print("\r{} / {}\t".format(ii, Nn), end='')
        idx = anchors_idx[xx]
        f1 = os.path.join(image_path, "MDS_"+str(idx)+".npy")
        img1 = np.load(f1)
        pp_dist = Parallel(n_jobs=-1, verbose=1)(delayed(_yp_dist)(yy, image_path, train_idx, img1, connectivity, **kwargs) for yy in range(Mm) )
        dist_mat[xx, :] = np.asarray(pp_dist).reshape(1, -1)
        ii += 1
    # response_dist = pairwise_distances(anchors_reponse, response, metric=metric)
    vmax = np.nanmax(dist_mat)
    np.nan_to_num(dist_mat, copy=False, nan=vmax)
    return dist_mat


def image_x_test(image_path, test_idx, anchors_idx, connectivity, **kwargs):
    Nn = len(test_idx)
    Mm = len(anchors_idx)
    dist_mat = np.zeros((Nn, Mm))
    verb = kwargs.get("verbose")
    ii = 1

    for xx in range(Nn):
        idx = test_idx[xx]
        f1 = os.path.join(image_path, "MDS_"+str(idx)+".npy")
        img1 = np.load(f1)
        for yy in range(Mm):
            if verb:
                print("\r{} / {}".format(ii, Nn*Mm), end='')
                ii += 1
            col = anchors_idx[yy]
            if idx == col:
                pp_dist = 0
            else:
                f2 = os.path.join(image_path, "MDS_"+str(col)+".npy")
                img2 = np.load(f2)
                pp_dist = image_distance.p2p_distance(img1, img2, connectivity,
                                             **kwargs)
            dist_mat[xx, yy] = pp_dist
        # padel_dist = pairwise_distances(test_padel, anchors_padel, metric=metric)
    vmax = np.nanmax(dist_mat)
    np.nan_to_num(dist_mat, copy=False, nan=vmax)
    return dist_mat

def test_y_image_distance():
    response = loaddata.load_GDSC_rna_response()['aucs']
    response = response.fillna(response.mean(axis=0))
    anch = select_anchor_idx(response, response, 100)
    dist_simple = simple_y_train(response, anch, "euclidean")
    folder = "../topological_regression_output/GDSC_auc_images_new"
    dist_img_1 = image_y_train(folder, response.index, anch, "Euclidean", verbose=1) / 48
    dist_img_2 = image_y_train_p(folder, response.index, anch, "Euclidean", verbose=1) / 48
    assert(np.allclose(dist_img_1, dist_img_2))
#%%

def _xp_Leven_dist(seq_a, series, idx):
    from Levenshtein import distance as Leven_distance
    seq_b = series.loc[idx]
    pp_dist = Leven_distance(seq_a, seq_b)
    return pp_dist

def Levenshtein_x_train_p(df, train_idx, anchors_idx, relative=True, verbose=True):
    """
    df is a pd.DataFrame, with a "Seqs" column.
    df contains all the samples. The distances are
    Multi-thread computing using joblib.

    Pass train_idx = None to calculate all sample no matter train or test

    """
    if train_idx is None:
        train_idx = df.index
    Nn = len(train_idx)
    Mm = len(anchors_idx)
    dist_matr = np.zeros((Nn, Mm))
    seq_ser = df["Seqs"]

    for jj in range(Mm):
        if verbose:
            print("\r Levenshtein distance: {} / {}\t".format(jj+1, Mm), end='')
        each_seq = seq_ser.loc[anchors_idx[jj]]
        pp_dist = Parallel(n_jobs=-1)(delayed(_xp_Leven_dist)(each_seq, seq_ser, ii) for ii in train_idx)
        dist_matr[:, jj] = pp_dist

    if relative:
        dist_matr = dist_matr / dist_matr.max()
    if verbose:
        print("\n")

    return dist_matr

def _xp_SW_score(base_str, seq_ser, submtr, idx, **kwargs):
    # **kwargs can be passed, in later. Changing gap cost, etc..
    query = SW(base_str, substitution_matrix=submtr, score_only=True,
                        protein=True, suppress_sequences=True, **kwargs)
    seq_b = seq_ser.loc[idx]
    aln = query(seq_b)
    d = aln.optimal_alignment_score
    return d

def _xp_subs_score(base_str, seq_ser, submtr, idx, **kwargs):
    # base str is the anchor. 
    val = 0
    seq_b = seq_ser.loc[idx]
    d=difflib.Differ()
    diff = d.compare(base_str, seq_ser)
    char_diff = [x for x in diff if not x.startswith(' ')]
    for each_mut in char_diff:
        if each_mut.startswith('-'):
            old = each_mut[-1]
            val -= submtr[old][old]
        if each_mut.startswith("+"):
            new = each_mut[-1]
            val += submtr[old][new]
    return val
    
    
    
def SW_x_train_p(df, train_idx, anchors_idx, submtr=None, relative=True, verbose=True, alg='SW', **kwargs):

    """
    Provide a entry for seq distance. 懒得改了，就用同一个入口吧。
    SW for Smith Waterman.
    sum_subs for calculate the decrease of align scores. And take the sum. 
    """
    if train_idx is None:
        train_idx = df.index
    Nn = len(train_idx)
    if isinstance(anchors_idx, str):
        anchors_idx = [anchors_idx]
    Mm = len(anchors_idx)
    simi_matr = np.zeros((Nn, Mm))
    seq_ser = df["Seqs"]

    if submtr is None:
        submtr = blosum50
    else:
        submtr = submtr.lower()
        submtr = load_subs_mtrx(submtr)

    for jj in range(Mm):
        if verbose:
            print("\r Smith-Waterman distance: {} / {}\t".format(jj+1, Mm), end='')
        each_seq = seq_ser.loc[anchors_idx[jj]]
        # query = SW(each_seq, substitution_matrix=submtr, score_only=True,
        #                protein=True, suppress_sequences=True)
        # SW suery object cannot be pickeled.
        if 0:          # for debug:
            print("debug mode non-para ver")
            pp_dist = []
            for each_idx in train_idx:
                try:
                    pp_dist.append(_xp_SW_score(each_seq, seq_ser, submtr, each_idx) )
                except:
                    print(each_idx)
                    print(seq_ser[each_idx])
                    raise
        # Code continue from here:
        if alg=='SW':
            pp_dist = Parallel(n_jobs=-1)(delayed(_xp_SW_score)(each_seq, seq_ser, submtr, each_idx) for each_idx in train_idx)
        elif alg=='sum_subs':
            pp_dist = Parallel(n_jobs=-1)(delayed(_xp_subs_score)(each_seq, seq_ser, submtr, each_idx) for each_idx in train_idx)
        simi_matr[:, jj] = pp_dist
    
    # 这个dist matr 都是相似度。不是距离。
    # simi_matr = simi_matr.max() - simi_matr  # added 4/1/21. 
    # TODO: 有时在HPC 会把anchors 拆开。这个写法会按被拆的anchor 统计最大值。不准确。后面要把倒置挪到再上一层

    if relative:
        simi_matr = simi_matr / simi_matr.max()
    if verbose:
        print("\n")

    return simi_matr

def test_Levenshtein_p():
    from Levenshtein import distance as Leven_distance
    test_strs = ["QEGHHI", "QEGHII", "CHILKP", "LR"]
    value = [1, 2, 3, 4]
    test_df = pd.DataFrame({"Seqs": test_strs, "Vals": value}, index=list('abcd'))
    d = Levenshtein_x_train_p(test_df, list('abcd'), ['a', 'c'], False)
    
    submtr = load_subs_mtrx("blosum50")
    
    d0 = [_xp_SW_score("QEGHII", test_df["Seqs"], submtr, ii) for ii in 'abcd']

    assert(d[3, 1] == Leven_distance("LR", "QEGHII"))
    
    

#%%
if __name__ == "__main__":
    test_simple_x_dist()
    # test_Levenshtein_p()
    raise
    # test_isomap_dist()
    test_y_image_distance()
    raise
    padel, response = loaddata.load_NCI_responses_padel()
    anchors_idx = select_anchor_idx(padel, response, 50)

    d = isomap_y_train(response, anchors_idx, "euclidean", n_neighbors=5)


    t = test_pca_anchor()
    print(t)

    # Test PCA distance:
    dx, pcax = pca_x_train(padel, anchors_idx, 3, metric='euclidean')
    dy = pca_y_train(response, anchors_idx, 3, "euclidean")
    xt = padel.iloc[-100:, :]
    dxt = pca_x_test(pcax, padel, xt, anchors_idx, "euclidean")
    assert(dxt.shape==(100, 50))




