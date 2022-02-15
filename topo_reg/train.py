"""

Train from precomputed distances.

"""
import numpy as np
import pandas as pd
import joblib
from topo_reg.args import TarinPrecompuArgs, TrainArgs
from topo_reg.io import check_path_exists, load_int_dict_from_json, read_df, read_df_list
from topo_reg.sklearn_model import get_arbitary_model, get_model
from topo_reg.calculate_distance import simple_y_train


def train_precomputed(args: TarinPrecompuArgs):
    # Prepare target columns
    target_df = read_df(args.target_file)
    if args.targets is None:
        target = target_df
    else:
        target = target_df[args.targets]
    # Load precomputed distance matrix
    distance = read_df_list(args.distance_path)
    if args.is_similarity:
        distance = distance.max().max() - distance
    # Prepare sklearn model
    if args.model_params is None:
        params = {}
    else:
        params = load_int_dict_from_json(args.model_params)
    if '.' in args.model:
        mdl = get_arbitary_model(args.model, params)
    else:
        mdl = get_model(args.model, params)
    # load training index
    if args.index_json is None:
        train_idx = distance.index
        anchors_idx = distance.index
    else:
        index_json = load_int_dict_from_json(args.index_json)
        train_idx = pd.Index(index_json['train_idx'])
        anchors_idx = pd.Index(index_json['anchors_idx'])

    # check index are matched.
    assert(train_idx.isin(distance.index).all()), "Train index doesn't match distances index"
    assert(train_idx.isin(target_df.index).all()), "Train index doesn't match target df"

    dist_x_train = distance.loc[train_idx].values
    dist_y_train = simple_y_train(target, anchors_idx, args.y_dist_metric, train_idx=train_idx)
    # modelling
    try:
        mdl.fit(dist_x_train, dist_y_train.T)
    except MemoryError:
        dist_x_train = dist_x_train.astype(np.float16)
        dist_y_train = dist_y_train.astype(np.float16)
        mdl.fit(dist_x_train, dist_y_train.T)

    if not args.model_save_path is None:
        check_path_exists(args.model_save_path)
        joblib.dump(mdl, args.model_save_path)

    return mdl


