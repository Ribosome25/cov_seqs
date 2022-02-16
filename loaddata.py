# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:42:57 2020

@author: ruibzhan
"""
#%%
import numpy as np
import pandas as pd
#%%

#%%
# The orders of seqs in DFs are matched. 
# VHSE used I + number index. 
def load_Sci_embeddings(pooling='mean', dataset='fitness'):
    if 'fit' in dataset:
        df = pd.read_parquet("data/fitness_embeddings/Seqs_Fitness.parquet")
        folder = 'fitness_embeddings'
    elif 'express' in dataset:
        df = pd.read_parquet("data/expression_embeddings/Seqs_Expression.parquet")
        folder = 'expression_embeddings'
    else:
        raise ValueError("{}: dataset unknown".format(dataset))

    if 'av' in pooling.lower() or 'mean' in pooling.lower():
        embd = pd.read_parquet("data/{}/LSTM_Embeddings_Mean.parquet".format(folder))
        # embd = pd.read_parquet("data/{}/Embeddings_Mean_expression.parquet".format(folder))

    elif 'sep' in pooling.lower() or 'last' in pooling.lower():
        embd = pd.read_parquet("data/{}/LSTM_Embeddings_Last.parquet".format(folder))
    elif 'cscs' in pooling.lower():
        embd = pd.read_parquet("data/{}/CscsScore.parquet".format(folder))[['sem_change', 'cscs']]
    else:
        raise ValueError("{}: This pooling method is not calculated. ".format(pooling))
    return embd, df

def load_Bert_embeddings(pooling='average', dataset='fitness'):
    if 'fit' in dataset:
        df = pd.read_parquet("data/fitness_embeddings/Seqs_Fitness.parquet")
        folder = 'fitness_embeddings'
    elif 'express' in dataset:
        df = pd.read_parquet("data/expression_embeddings/Seqs_Expression.parquet")
        folder = 'expression_embeddings'
    else:
        raise ValueError("{}: dataset unknown".format(dataset))
        
    if 'av' in pooling.lower() or 'mean' in pooling.lower():
        embd = pd.read_parquet("data/{}/Bert_Embeddings_MeanPooling.parquet".format(folder))
    elif 'cls' in pooling.lower() or 'first' in pooling.lower():
        embd = pd.read_parquet("data/{}/Bert_Embeddings_ClsPooling.parquet".format(folder))
    elif 'sep' in pooling.lower() or 'last' in pooling.lower():
        embd = pd.read_parquet("data/{}/Bert_Embeddings_LastPooling.parquet".format(folder))
    else:
        raise ValueError("{}: This pooling method is not calculated. ".format(pooling))
    return embd, df

def load_VHSE():
    # Only for fitness dataset. expression 的 另外写一个。
    # 必须得对齐sci bert 和 这个df 的index顺序.
    _, df_sci = load_Sci_embeddings()
    # del _
    embd = pd.read_parquet("data/fitness_embeddings/from_HPC_VHSE_embedding.parquet", engine='fastparquet')
    df = pd.read_parquet("data/fitness_embeddings/fullset_df_to_HPC.parquet", engine='fastparquet')
    concat = pd.merge(df_sci, df.reset_index(), on='Seqs').set_index('index')

    return embd.reindex(concat.index), df.reindex(concat.index)

def load_VHSE_Expression():
    embd = pd.read_parquet("data/expression_embeddings/from_HPC_VHSE_embedding_expression.parquet", engine='fastparquet')
    df = pd.read_parquet("data/expression_embeddings/Seqs_Expression.parquet", engine='fastparquet')
    return embd, df

def load_others(which='intersection_gisaid_starr'):
    if ('intersec' in which and \
        'gisaid' in which and \
        'starr' in which):
        return pd.read_parquet("../../Datasets/CoV_virus_seq/gisaid_itc_starr.parquet", engine='fastparquet')

#%%
if __name__ == "__main__":
    pass