"""

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ref_models.args import VHSE_CNN_predict_Args, VHSE_CNN_training_Args
from ref_models.cnn_model import build_VHSE_CNN
from topo_reg.io import check_path_exists, load_int_dict_from_json, read_df_list, save_df
from sklearn.model_selection import train_test_split
from tensorflow import keras


def predict_VHSE_CNN(args: VHSE_CNN_predict_Args):
    model = keras.models.load_model(args.model_path)
    x_df = read_df_list(args.x_test)
    x_test = x_df.values.reshape(len(x_df), -1, 8)
    prediction = pd.DataFrame(model.predict(x_test), index=x_df.index)

    save_df(prediction, args.save_path)

