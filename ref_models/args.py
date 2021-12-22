"""
Ref model arguments

"""

from typing import List
from tap import Tap


class MPI_VHSE_Args(Tap):
    df_path: str  # DF with a 'Seqs' column
    save_path: str  # embed VHSE vectors


class VHSE_CNN_training_Args(Tap):
    max_length: int=1273  # length of seqeunces must be given beforehand
    x_train: List  # x files for training
    target_file: List  # target file contains the target column
    targets: List  # target columns
    model_params: str=None  # a json file
    save_dir: str  # where to save the model, training history etc. 
    plot_history: bool=False  # if True, save a png in the same folder
    cross_valid: bool=False
    seed: int=2021


class VHSE_CNN_predict_Args(Tap):
    x_test: List 
    model_path: str
    save_path: str=None  # where to save the predictions. If None, predictions will be saved to the same dir as model
    target_file: List=None  # pass the target df to get reg scores
    targets: List=None