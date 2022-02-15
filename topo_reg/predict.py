"""

"""


from platform import dist
import joblib
import pandas as pd
from topo_reg.io import check_is_file, check_path_exists, load_int_dict_from_json, read_df, read_df_list
from topo_reg.args import PredictPrecompuArgs
from topo_reg import reconstruct
from sklearn.base import BaseEstimator

def predict_precomputed(args: PredictPrecompuArgs, mdl: BaseEstimator=None):
    """
    mdl parameter is used when we don't want to save and load the model. 
    """
    distance = read_df_list(args.distance_path)
    if args.is_similarity:
        distance = distance.max().max() - distance

    target_df = read_df(args.target_file)
    if args.targets is None:
        target = target_df
    else:
        target = target_df[args.targets]
    if args.index_json is None:
        anchors_idx = target.index
        test_idx = distance.index
    else:
        index_json = load_int_dict_from_json(args.index_json)
        test_idx = pd.Index(index_json['test_idx'])
        anchors_idx = pd.Index(index_json['anchors_idx'])

    # >>>> 大test 样本 在rbf reconstruction 中会内存溢出。临时解决方法。以后再改。
    if len(test_idx) > 30000:
        further_concat = []
        for test_idx in [test_idx[i:i + 30000] for i in range(0, len(test_idx), 30000)]:
        
            dist_test = distance.loc[test_idx]
            # model prediction
            if mdl is None:
                mdl = joblib.load(args.model_path)
            print(str(mdl))
            dist_array_test = mdl.predict(dist_test.values).T

            predict_values = []
            for each_gamma in args.rbf_gamma:
                each_gamma = float(each_gamma)
                response_array_r_t = reconstruct.rbf(dist_array_test, target, anchors_idx, each_gamma, False)
                predict_values.append(pd.DataFrame(response_array_r_t, index=test_idx,
                        columns=[str(ii) + "_RBF_" + str(each_gamma) for ii in target.columns]))

            for each_k in args.knn:
                each_k = int(each_k)
                response_array_k_t = reconstruct.knn(dist_array_test, target, anchors_idx, knn=each_k)
                predict_values.append(pd.DataFrame(response_array_k_t, index=test_idx,
                        columns=[str(ii) + "_KNN_" + str(each_k) for ii in target.columns]))

            check_path_exists(args.save_path)
            assert(check_is_file(args.save_path))
            part_result = pd.concat(predict_values, axis=1)
            further_concat.append(part_result)
        pd.concat(further_concat, axis=0).to_csv(args.save_path)
    # <<<<<<  else 后面是原版。
    else:
        dist_test = distance.loc[test_idx]
        # model prediction
        if mdl is None:
            mdl = joblib.load(args.model_path)
        print(str(mdl))
        dist_array_test = mdl.predict(dist_test.values).T

        predict_values = []
        for each_gamma in args.rbf_gamma:
            each_gamma = float(each_gamma)
            response_array_r_t = reconstruct.rbf(dist_array_test, target, anchors_idx, each_gamma, False)
            predict_values.append(pd.DataFrame(response_array_r_t, index=test_idx,
                    columns=[str(ii) + "_RBF_" + str(each_gamma) for ii in target.columns]))

        for each_k in args.knn:
            each_k = int(each_k)
            response_array_k_t = reconstruct.knn(dist_array_test, target, anchors_idx, knn=each_k)
            predict_values.append(pd.DataFrame(response_array_k_t, index=test_idx,
                    columns=[str(ii) + "_KNN_" + str(each_k) for ii in target.columns]))

        check_path_exists(args.save_path)
        assert(check_is_file(args.save_path))
        pd.concat(predict_values, axis=1).to_csv(args.save_path)
