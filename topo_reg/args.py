"""
TAP Arguments

"""

from typing import List
from tap import Tap  # pip 

#%% Distances
class DistanceArgs(Tap):
    # train index 和 test index 没有本质区别
    # 只是因为要分别从两个文件里sample，又要同时放进mpi 才分开的
    # 如果是本地运算，不做precomputed，不需要分train 和test index。precomputed 需要
    df_path: List  # The paths to the df files. Files will be concat before sent to calc dist. Make sure this is a common col e.g. Seqs.
    x_columns: List = None  # None for all
    index_path: str  # A json file defining train_idx, anchors_idx, and test_idx
    test_index_path: str = None  # If given, test_idx will be overwritten by this one. 
    save_path: str  # File to save the computated distances\
    method: str = 'simple'  # simple distance, image distance, SW distance, etc..
    metric: str = 'euclidean'

class SW_Args(DistanceArgs):
    sub_matrix: str = 'blosum55'  # Which substitute matrix to be used when calculating S-W distance.

class ImageDistArgs(DistanceArgs):
    image_folder: str  # to be done

#%% Sample and anchors
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

class AnchorsArgs(Tap):
    """
    Set None to num_train or num_test to save all index as train/test.
    If num_train and num_anchors are both None, N of train index will be set 0, and test for all. 
    If given test_portion, it will do the train_test_split instead of select numbers. 

    """

    df_path: List  # The paths to the df files. Files will be concatenated.
    num_train: int = None  # How many are sampled as training sample. None for all. 
    test_portion: float = None  # Highest priority. The proportion of test set. Will overwrite nums if given. 
    num_anchors: int = 200  # How many are sampled (from training) as anchors. 
    save_path: str  # Where to save the json file. 
    seed: int = 2021  # Random seed
    def configure(self) -> None:
        self.add_argument('--num_test', type=none_or_int, default=0, nargs='?',
            help= "How many are sampled (outside training) as test. None for all.\
                 There can be only one None for num_train and num_test. OTW train will be overwritten.")


class JsonArgs(Tap):
    # combine two jsons
    train_json: str  # the json file defining the training index
    anchors_json: str  # the json file defines anchors index
    test_json: str  # the json file which contains test index
    save_path: str  # the json file path to be saved


class TrainArgs(Tap):
    model: str  # LR, Ridge, RF, GPR
    model_params: str=None  # path to a json. Ignored the .fit() params.
    target_file: str  # The df of target values for training.
    targets: List=None  # The target columns. If None, all columns will be targets.
    y_dist_metric: str='euclidean'  # distance metric for calc y distance. Will be passed to pairwise_distance()
    index_json: str=None  # path to the json which stores the index. If None, all will be considered training data
    model_save_path: str=None  # path to save the model

class TarinPrecompuArgs(TrainArgs):
    distance_path: List  # the path to the df.
    is_similarity: bool=False  # sometimes the precomputed is similarity not distance. WIll be subtracted by the max. 

class TrainRawArgs(TrainArgs):
    df_path: List  # The paths to the df files. Files will be concatenated.
    x_columns: List = None  # None for all
    x_dist_metric: str = 'euclidean'
    dist_save_path: str = None  # if None, do not save. 


    models: List=[]  # Because train from raw takes a long time. Fit multi models here.

"""
    num_train: int = None  # How many are sampled as training sample. None for all. 
    num_anchors: int = 200  # How many are sampled (from training) as anchors. 
    index_save_path: str  # Where to save the index json file. 
    seed: int = 2021  # Random seed
"""
class PredictArgs(Tap):
    target_file: str  # The df of target values for reconstruction.
    targets: List=None  # The target columns. If None, all columns will be targets.
    model_path: str  # The base regression model.
    rbf_gamma: List=[]
    knn: List=[]  # the number of K.
    save_path: str  # Where to write the predictions.

class PredictPrecompuArgs(PredictArgs):
    """
    In precompu prediction: model are passed. targets for reconstruct are provided.
    """
    distance_path: List  # a df.
    index_json: str # a json with 'test_idx': [...]
    is_similarity: bool=False  # sometimes the precomputed is similarity not distance. WIll be subtracted by the max. 

#%%
class ConjointTriadArgs(Tap):
    df_path: str  # path to the df contains 'Seqs' column
    save_path: str  # path to save the df


#%%  Pipeline Args
class ChemblPipelineArgs(Tap):
    path: str  # the working folder that contains the target and descs files
    metric: str='tanimoto'  # the default distance metric to use
    seed: int=2021  # random seed 
