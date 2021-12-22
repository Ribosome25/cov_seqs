# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:20:56 2021

@author: Ruibo
"""
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import pandas as pd
import os
import requests
from tqdm.auto import tqdm
import sys

def main(part):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)
    # sequences_Example = ["A E T C Z A O","S K T Z P"]  #TODO

    # data = pd.read_parquet("../viral-mutation/fitness_embeddings/Seqs_Fitness.parquet")
    # data = pd.read_parquet("../viral-mutation/expression_embeddings/Seqs_Expression.parquet")
    # data = pd.read_csv("/home/ruibzhan/Datasets/GISAID_0820/Seqs_w_meta_.csv")
    data = pd.read_parquet('/home/ruibzhan/Datasets/GISAID_1203/GISAID1203+omicron.parquet')

    if part == 1:
        data = data.iloc[:60000]
    elif part == 2:
        data = data.iloc[60000:]
    else:
        print("This is the test run.")
        data = data.iloc[:2000]

    Nn = len(data)
    seqs = data['Seqs']
    cols = ["F{}".format(x) for x in range(1024)]

    cls_pool = pd.DataFrame(np.zeros((Nn, 1024)), index=data.index, columns=cols)
    avg_pool = pd.DataFrame(np.zeros((Nn, 1024)), index=data.index, columns=cols)
    sep_pool = pd.DataFrame(np.zeros((Nn, 1024)), index=data.index, columns=cols)

    for batch_sequences in tqdm(np.array_split(seqs, 1000)):
        edited_batch_sequences = [re.sub(r"[UZOB]", "X", " ".join(sequence)) for sequence in batch_sequences]
        embedding = fe(edited_batch_sequences)
        embedding = np.array(embedding)

        # print(embedding.shape)
        cls_pool.loc[batch_sequences.index] = embedding[:, 0, :]
        avg_pool.loc[batch_sequences.index] = embedding.mean(axis=1)
        sep_pool.loc[batch_sequences.index] = embedding[:, -1, :]

        del embedding

    # cls_pool.to_parquet("./express_set/Bert_Embeddings_ClsPooling_part{}.parquet".format(part), compression='gzip')
    # avg_pool.to_parquet("./express_set/Bert_Embeddings_MeanPooling_part{}.parquet".format(part), compression='gzip')
    # sep_pool.to_parquet("./express_set/Bert_Embeddings_LastPooling_part{}.parquet".format(part), compression='gzip')
    # cls_pool.to_parquet("./GisAid_0820/Bert_Embeddings_ClsPooling_part{}.parquet".format(part), compression='gzip')
    # avg_pool.to_parquet("./GisAid_0820/Bert_Embeddings_MeanPooling_part{}.parquet".format(part), compression='gzip')
    # sep_pool.to_parquet("./GisAid_0820/Bert_Embeddings_LastPooling_part{}.parquet".format(part), compression='gzip')
    cls_pool.to_parquet("./GisAid_1203/Bert_Embeddings_ClsPooling_part{}.parquet".format(part), compression='gzip')
    avg_pool.to_parquet("./GisAid_1203/Bert_Embeddings_MeanPooling_part{}.parquet".format(part), compression='gzip')
    sep_pool.to_parquet("./GisAid_1203/Bert_Embeddings_LastPooling_part{}.parquet".format(part), compression='gzip')
    print(cls_pool.shape, avg_pool.shape, sep_pool.shape)

if __name__ == "__main__":
    main(0)
    main(1)
    main(2)
