# cov_seqs

Code for reproducing the results in *Zhang, et. al. Predicting binding affinities of emerging variants of SARS-CoV-2 using spike protein sequencing data: Observations, caveats and recommendations (Briefings in bioinformatics 2022, in press)* 

![protocol](figures/protocol.png)

## Clustering scatter plot and visualization

See visualization/GisAid1203.ipynb

![mds-variant](figures/mds-variant.png)

## Topological regression performance for DMS lab dataset

Refer to `model_CoV_seq.py` and `model_CoV_seq_clf`

Topo Reg using CNN as similarity metric: `model_CoV_CNN_topo`

## LSTM / BERT embedding generation

LSTM is modified from https://github.com/brianhie/viral-mutation. bin/combinatorial_fitness.py, bin/mutation.py and bin/cov.py are modified for our usage. The modified files are provided in `ref_models/science2021`. 

BERT embedding is generated with `ref_models/ProtBert/ProtBert.py`.

The script for modeling and performance is provided in `scripts/model_CoV_seq.py`

Parameter search is in `model_NLP_fc_gridsearch` and `model_VHSE_CNN_gridsearch_clf` .

## VHSE - CNN

Model output, performance: `model_CoV_VHSE_CNN` and `model_CoV_VHSE_CNN_clf`

Grid search parameter tuning: `model_VHSE_CNN_gridsearch_clf` 

Random search parameter tuning: `keras_tuner_CNN_run.py`

## Data preprocessing

Including VHSE encoding, canculate distance, etc. MPI and HPC are required in some methods. 

See `ref_models` and `scripts`
