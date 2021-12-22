"""
Script to run for predicting GISAID

"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import Ridge
from topo_reg.io import read_df_list
#%%
f_name = "output/Sci&Bert_refmodel_GISAID.csv"
#%%
y = pd.read_parquet("data/fitness_embeddings/Seqs_Fitness.parquet", engine='fastparquet')
y_train = y['fitness']

# y_test = pd.read_csv(r"data/GisAid/Seqs_w_meta_.csv", index_col=0)
y_test = pd.read_parquet("data/GisAid/GISAID1203+omicron.parquet")
y_test['Seqs'] = y_test['Seqs'].str.strip('*')
outputs = []
sci_f_list = [
    "data/GisAid/GisAid_1203/Embeddings_Mean_part1.parquet",
    "data/GisAid/GisAid_1203/Embeddings_Mean_part2.parquet",
]
bert_f_list = [
    "data/GisAid/GisAid_1203/Bert_Embeddings_MeanPooling_part1.parquet",
    "data/GisAid/GisAid_1203/Bert_Embeddings_MeanPooling_part2.parquet",
]

for d in ['LSTM', 'Bert']:
    if d == 'LSTM':
        x_train = pd.read_parquet("data/fitness_embeddings/Embeddings_Mean.parquet", engine='fastparquet')
        x_test = read_df_list(sci_f_list)
        x_test_meta = read_df_list(["data/GisAid/GisAid_1203/Seqs_1203_sem_change_part1.parquet", "data/GisAid/GisAid_1203/Seqs_1203_sem_change_part2.parquet"])
        x_test.index = x_test_meta['Seqs']
        x_test = x_test.reindex(y_test['Seqs'])
        x_test.index = y_test.index

    elif d == 'Bert':
        x_train = pd.read_parquet("data/fitness_embeddings/Bert_Embeddings_MeanPooling.parquet", engine='fastparquet')
        x_test = read_df_list(bert_f_list)

    for m in ['RF', 'Ridge']:
        if m == 'RF':
            mdl = RFR(max_depth=4, n_jobs=-1, verbose=1)
        elif m == 'Ridge':
            mdl = Ridge()

        mdl.fit(x_train, y_train)
        pred = mdl.predict(x_test)

        outputs.append(pd.Series(pred, name=d+" "+m, index=y_test.index))
        print(d+m)
pd.concat(outputs, axis=1).to_csv(f_name)
