# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:30:03 2021

@author: Ruibo

11/29/21  补实验。三个fitness data set 的cscs score LR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from myToolbox.Metrics import sextuple
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

cscs = pd.read_parquet("data/fitness_embeddings/CscsScore.parquet")

RANDOM_SEED = 20
n_anchors = 500
# 11/19/21 run 三个Frac。距离都是Sum of Euclidean along seqeunce residuals
FRAC = 0.05

df = cscs.sample(frac=FRAC, random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = train_test_split(df[['sem_change', 'cscs']], df['fitness'], test_size=0.2, random_state=RANDOM_SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

#%%
mdl = LR()
scaler = MinMaxScaler()
xx = scaler.fit_transform(X_train)
mdl.fit(xx, y_train)
pred = mdl.predict(scaler.transform(X_test))

mdl.fit(X_train, y_train)
pred = mdl.predict(X_test)
#%%
perf = sextuple(y_test, pred, False)
print(FRAC, perf)
