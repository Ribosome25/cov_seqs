"""
All IO functions
"""

import json
from typing import List, Optional
import pandas as pd
import os
import warnings
import math
import pickle

#%%
def check_input_data_df(df: pd.DataFrame):
    """
    Check headers contain "Seqs"
    """
    assert("Seqs" in df.columns), "Cannot find Seqs column in the input file."
    return df


def check_path_exists(output_path: str):
    """
    if output_path is a dir, check the dir exist, or mkdir. 
    if output_path is a file, check the folder exists, or mkdir.
    """
    if output_path is None:
        return None
    if os.path.isdir(output_path):
        # if this is a folder that already exsits:
        return output_path
    else:
        # if there is no extension in file name, consider it as a dir.
        # check if it exsits, and if not, create the dir.
        path, ext = os.path.splitext(output_path)

        if ext == '':
            # This is a dir or a file without ext.
            if not os.path.exists(path) or os.path.isfile(path):
                os.makedirs(path)
        else:
            # This is a file name.
            if os.path.exists(output_path):
                warnings.warn("{} already exists. Will be over-written.".format(output_path))
            dir, f_name = os.path.split(output_path)
            if not os.path.exists(dir):
                # if the dir not exsit.
                os.mkdir(dir)
            elif os.path.isfile(path):
                # if it is a file with the same name.
                os.mkdir(dir)

        return output_path


def check_is_file(path: str or List) -> bool:
    """
    Check if the path(s) is a file or a dir.
    If a list is passed, it's not a file.
    Check if the path has a extention. If not it'll be considered a dir.
    """
    if not isinstance(path, str):
        # if path is a list, or something, return a False. It's not a file.
        return False
    file, ext = os.path.splitext(path)
    return ext != ''

def parent_dir(path: str) -> str:
    if check_is_file:
        return os.path.split(path)[0]
    else:
        return path
#%%
def read_df(path: str, convert_index=True) -> pd.DataFrame:
    """
    Read single df. 
    If convert_index is True, this function will search for that if the index is RangeIndex. 
    If yes, it will be converted to I{} format. 
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, engine='fastparquet')
    elif path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    elif path.endswith(".tsv"):
        df = pd.read_table(path, index_col=0)
    elif path.endswith(".pickle") or path.endswith('.pkl'):
        with open(path, 'rb') as f:
            df = pickle.load(f)
    elif path.endswith(".json"):
        dict = json.load(path)
        df = pd.DataFrame(dict)
    elif path.endswith(".txt"):
        df = pd.read_table(path, index_col=0)
    else:
        raise ValueError("Not supported format: " + path)

    if convert_index:
        if isinstance(df.index, pd.RangeIndex):
            df.index = ["I{}".format(ii) for ii in range(len(df))]
        if isinstance(df.columns, pd.RangeIndex):
            df.columns = ["I{}".format(ii) for ii in range(df.shape[1])]

    return df


def read_df_list(paths: List) -> pd.DataFrame:
    dfs = []
    for each_path in paths:
        dfs.append(read_df(each_path))
    cdf = pd.concat(dfs)
    if sum(cdf.index.duplicated()) > 0:
        print(cdf.index[cdf.index.duplicated()])
        # raise ValueError("Duplicated index are found in the DFs.")
        print("Duplicated index are found in the DFs.")
    return cdf


def save_df(obj: pd.DataFrame, save_path:str, tag=None):
    check_path_exists(save_path)
    if tag is not None:
        tag = "_" + tag.strip("_")
    else:
        tag = ""
    if save_path.endswith(".parquet"):
        if isinstance(obj.index, pd.RangeIndex):
            obj.index = ["F{}".format(ii) for ii in range(obj.shape[1])]
        save_path = save_path.replace(".parquet", tag + ".parquet")
        obj.to_parquet(save_path, compression='gzip')
    elif save_path.endswith(".csv"):
        save_path = save_path.replace(".csv", tag + ".csv")
        obj.to_csv(save_path)
    else:
        raise ValueError("Unkonwn save type.")


#%%
def log_to_csv(message, path: str) -> None:
    """Write (append) a list to csv. """
    if isinstance(message, str):
        message = [message]
    check_path_exists(path)
    if not check_is_file(path):
        path = os.path.join(path, "log.csv")
    string = "\n{}" + ",{}"*(len(message)-1)
    with open(path, 'a') as file:
        file.write(string.format(*message))


#%%
def float_to_int(x):
    # int becomes float when loaded by json
    if not isinstance(x, float):
        return x
    elif math.isnan(x) or math.isinf(x):
        return x
    else:
        if int(x) == x:
            return int(x)
        else:
            return x

def load_int_dict_from_json(path: str) -> dict:
    # automatic convert floated int back
    with open(path, 'r') as f:
        origin = json.load(f)
    processed = {key: float_to_int(origin[key]) for key in origin}
    return processed

#%%
def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.
    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.
    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None

if __name__ == "__main__":
    # test codes
    import numpy as np
    x = {'a': 10.0, 'b': 'tt', 'c': np.nan}
    processed = {key: float_to_int(x[key]) for key in x}
    print(processed)
