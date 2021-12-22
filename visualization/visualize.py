# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:43:59 2021

@author: ruibzhan
"""
import random, pickle
import numpy as np
import pandas as pd
from Levenshtein import distance
from itertools import combinations
from sklearn.manifold import MDS, Isomap
import matplotlib.pyplot as plt
import seaborn as sns
from skbio.sequence import Protein
from skbio.alignment import global_pairwise_align_protein as nw
from skbio.alignment import StripedSmithWaterman as SW
def two_d_eq(xy):
    # xy is N x 2 xy cordinates, returns eq-xy on [0,1]
    xx_rank = np.argsort(xy[:,0])
    yy_rank = np.argsort(xy[:,1])
    eq_xy = np.full(xy.shape,np.nan)
    for ii in range(xy.shape[0]):
        xx_idx = xx_rank[ii]
        yy_idx = yy_rank[ii]
        eq_xy[xx_idx,0] = ii * 1/len(xy)
        eq_xy[yy_idx,1] = ii * 1/len(xy)
    return eq_xy

def sample_from_dict(d, sample=10):
    if sample is None:
        sample = len(d)
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))

def process_dict_to_df(seqs_dict):
    dic = {"Seqs": [], "Dataset": [], "Host":[], "Group":[], "Strain":[]}

    for each in seqs_dict.keys():
        dic["Seqs"].append(str(each))
        dic["Dataset"].append(seqs_dict[each][0]['dataset'])
        strain = seqs_dict[each][0]['strain']
        dic["Strain"].append(strain)
        if (strain.lower().startswith('hcov-19') or strain.startswith("SARS-CoV-2")):
            dic["Host"].append(seqs_dict[each][0]['host'].capitalize() + "(hCoV19)")
        else:
            dic["Host"].append(seqs_dict[each][0]['host'].capitalize())
            # 为什么有时一个seq 对应多个meta?
        dic["Group"].append(seqs_dict[each][0]['group'])

    return pd.DataFrame(dic)

def calculate_dist_from_df(seq_df, relative=True, verbose=False):
    Nn = len(seq_df)
    max_len = 0
    dist_matr = np.diag(np.zeros(Nn))
    seq = seq_df['Seqs'].values
    disp_i = 0
    for ii, jj in combinations(range(Nn), 2):
        if verbose and disp_i != ii:
            disp_i = ii
            print(disp_i)
        stra = seq[ii]
        strb = seq[jj]
        d = distance(stra, strb)
        dist_matr[ii, jj] = d
        dist_matr[jj, ii] = d

    if relative:
        dist_matr = dist_matr / dist_matr.max()
    return dist_matr

def NW_distance_from_df(seq_df, submtr=None, gap_cost=11, relative=True, verbose=False):
    Nn = len(seq_df)
    dist_matr = np.zeros((Nn, Nn))
    seq = [Protein(x) for x in seq_df['Seqs'].values]
    for ii in range(Nn):
        if verbose:
            print("\r {} / {}".format(ii, Nn), end='')
        for jj in range(ii, Nn):
            aln = nw(seq[ii], seq[jj],
                     gap_open_penalty=gap_cost, gap_extend_penalty=1,
                     substitution_matrix=submtr)
            d = aln[1]
            dist_matr[ii, jj] = d
            dist_matr[jj, ii] = d
    if verbose:
        print("Done.")

    dist_matr = dist_matr.max() - dist_matr  # Matching score is a similarity.

    if relative:
        dist_matr = dist_matr / dist_matr.max()
    return dist_matr

def SW_distance_from_df(seq_df, submtr=None, relative=True, verbose=True, **kwargs):
    """

    :param **kwargs:
        gap_open_penalty, default is 5.
        gap_extend_penalty , default is 2.
        mask_auto, default is True (setting mask length to 1/2 seq length.)
    :type **kwargs: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    Nn = len(seq_df)
    dist_matr = np.zeros((Nn, Nn))
    seq = list(seq_df['Seqs'].values)
    if submtr is None:
        from substitution_matrix import blosum50
        submtr = blosum50
    for ii in range(Nn):
        if verbose:
            print("\rSmith-Waterman distance {} / {}".format(ii, Nn), end='')
            query = SW(seq[ii], substitution_matrix=submtr, score_only=True,
                       protein=True, suppress_sequences=True)
        for jj in range(ii, Nn):
            aln = query(seq[jj])
            d = aln.optimal_alignment_score
            dist_matr[ii, jj] = d
            dist_matr[jj, ii] = d
    if verbose:
        print("Done.")

    dist_matr = dist_matr.max() - dist_matr  # Matching score is a similarity.

    if relative:
        dist_matr = dist_matr / dist_matr.max()
    return dist_matr
#%%
def process_location_names(name: str) -> str:
    global CN_CITIES
    if name == "USA":
        return "United States"
    if name == "DRC":
        return "Democratic Republic of the Congo"
    if name in ["Guangdong", "Henan", "Sichuan"]:
        return "China"
    if name in CN_CITIES.index:
        return "China"
    if name in ["England", "Scotland", "Wales", "Northern Ireland"]:
        return "United Kingdom"
    else:
        return name.replace("_", " ")

def load_countries():
    import geonamescache
    gc = geonamescache.GeonamesCache()
    countries = pd.DataFrame(gc.get_countries()).T.set_index('name')
    return countries

def find_continent_by_country(country):
    global COUNTRIES
    dic = {"EU": "Europe",
           "AS": "Asia",
           "NA": "North America",
           "AF": "Africa",
           "SA": "South America",
           "OC": "Oceania",
           "AN": "Antarctica"}
    try:
        x = COUNTRIES.loc[country]
        cont = x['continentcode']
    except KeyError:
        print(country)
        return "Others"
    return dic[cont]

# import process_geonames
#%%
if __name__ == "__main__":
    with open('./data/cov_seqs.pickle', 'rb') as file:
        seqs = pickle.load(file)

    # CN_CITIES = pd.read_csv("./data/cn_cities.csv", index_col=0)
    # COUNTRIES = load_countries()

    selected_seqs = sample_from_dict(seqs, 100)
    # selected_seqs = seqs
    df = process_dict_to_df(selected_seqs)
    dist = SW_distance_from_df(df)
    raise
    # cov_df = df.loc[df["Host"] == "Human(hCoV19)"]
    # sub_group = []
    # higher_group = []
    # for each_idx, each_item in cov_df.iterrows():
    #     ds = each_item['Dataset']
    #     if ds == "nih":
    #         sub_group.append("NIH")
    #         higher_group.append("NIH")
    #     elif ds == "gisaid":
    #         location = each_item['Strain'].split("/")[1]
    #         sub_group.append(location)
    #         country_name = process_location_names(location)
    #         higher_group.append(find_continent_by_country(country_name))

    # cov_df["Strand_group"] = sub_group
    # cov_df["Higher_group"] = higher_group
    cov_df = process_geonames.process_cov19_df(df)
    with open("Levenshtein_results.pickle", 'rb') as f:
        _, dist = pickle.load(f)
    dist_df = pd.DataFrame(dist, index=df.index, columns=df.index)
    cov_dist = dist_df.loc[cov_df.index, cov_df.index]
    raise
    xy_cov19_mds = MDS(dissimilarity='precomputed').fit_transform(cov_dist)
    xy_cov19_mds = two_d_eq(xy_cov19_mds)
    plt.figure(figsize=(16, 12))
    sns.set_palette(sns.color_palette("tab10"))
    sns.scatterplot(x=xy_cov19_mds[:, 0], y=xy_cov19_mds[:, 1], hue=cov_df['Higher_group'], s=50)
    plt.title("Sequence proximity of coronavirus spike proteins || Levenshtein distance - MDS")
    raise
    #%%
    # dist = calculate_dist_from_df(df)
    aas = list("*ACBEDGFIHKMLNQPSRTWVYXZ")
    sub_matrx = pd.DataFrame(2 * np.diag(np.ones(24)) - 1, index=aas, columns=aas).to_dict()


    dist = NW_distance_from_df(df, verbose=1)
    dim_redu = MDS(dissimilarity='precomputed')
    xy = dim_redu.fit_transform(dist)

    plot = sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=df['Host'])
    h,l = plot.axes.get_legend_handles_labels()
    plot.axes.legend_.remove()
    plot.figure.legend(h,l, ncol=3, loc='center left', bbox_to_anchor=(1, 0.5))
