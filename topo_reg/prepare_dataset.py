"""
Prepare the data for MPI, etc. 

"""

import pandas as pd
import json
from topo_reg.args import AnchorsArgs, JsonArgs
from topo_reg.io import check_path_exists, load_int_dict_from_json, read_df_list
from sklearn.model_selection import train_test_split


def sample_anchors(args: AnchorsArgs):
    df = read_df_list(args.df_path)
    # if isinstance(df.index, pd.RangeIndex):
    #     df.index = ["I{}".format(ii) for ii in range(len(df))]

    if args.test_portion is not None:
        x_train, x_test = train_test_split(df, test_size=args.test_portion, random_state=args.seed)
        train_idx = x_train.index
        test_idx = x_test.index
        if args.num_anchors is None:
            num_anchors = len(df) - len(test_idx)
        else:
            num_anchors = min(args.num_anchors, len(df) - len(test_idx))
        anchors_idx = df.drop(index=test_idx).sample(num_anchors, random_state=args.seed).index
    else:
        if args.num_train is None:
            args.num_train = len(df)
        if args.num_test is None:
            print("Test index only.")
            args.num_train = 0
            args.num_test = len(df)

        train_idx = df.sample(args.num_train, random_state=args.seed).index
        test_idx = df.drop(index=train_idx).sample(args.num_test, random_state=args.seed).index
        if args.num_anchors is None:
            num_anchors = len(df) - len(test_idx)
        else:
            num_anchors = min(args.num_anchors, len(df) - len(test_idx))
        # anchors_idx = df.loc[train_idx].sample(num_anchors, random_state=args.seed).index  # anchors 也不必要是training 里的。
        anchors_idx = df.drop(index=test_idx).sample(num_anchors, random_state=args.seed).index

    output = {
        'train_idx': train_idx.tolist(),
        'anchors_idx': anchors_idx.tolist(),
        'test_idx': test_idx.tolist(),
    }

    check_path_exists(args.save_path)
    with open(args.save_path, 'w') as f:
        json.dump(output, f)

    return output


def combine_index_jsons(args: JsonArgs):
    tr = load_int_dict_from_json(args.train_json)['train_idx']
    ak = load_int_dict_from_json(args.anchors_json)['anchors_idx']
    tst = load_int_dict_from_json(args.test_json)['test_idx']
    output = {
        'train_idx': tr,
        'anchors_idx': ak,
        'test_idx': tst
    }
    check_path_exists(args.save_path)
    with open(args.save_path, 'w') as f:
        json.dump(output, f)

