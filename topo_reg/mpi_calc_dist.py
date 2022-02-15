"""
Function to calculate distances.


"""
import pandas as pd
from mpi4py import MPI
from topo_reg.mpi_util import scatter_list_to_processors, receive_from_processors_to_list
from topo_reg.args import SW_Args
from topo_reg.io import read_df_list, check_input_data_df, load_int_dict_from_json, check_path_exists, save_df
from topo_reg import calculate_distance


def calc_SW_distance_mpi(args: SW_Args):
    df = read_df_list(args.df_path)
    df = check_input_data_df(df)
    idx_dict = load_int_dict_from_json(args.index_path)
    if args.test_index_path is not None:
        test_idx_dict = load_int_dict_from_json(args.test_index_path)
        idx_dict['test_idx'] = test_idx_dict['test_idx']

    # want to do both at one time
    # train and test can not have same ids
    train_idx = list(idx_dict['train_idx'])
    anchors_idx = list(idx_dict['anchors_idx'])
    test_idx = list(idx_dict['test_idx'])
    both_idx = train_idx + test_idx

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    n_processors = comm.Get_size()

    if my_rank == 0:
        print("Processors found: ", n_processors)
        scatter_list_to_processors(comm, anchors_idx, n_processors)
        lst = receive_from_processors_to_list(comm, n_processors)
        result = pd.concat(lst, axis=1)

        # Note: collected are similarity scores. 
        # has to collect then reverse. Have to mod the calculate dist py accordingly. 
        result = result.max() - result

        # 其实也不用分开存。每一步都在调用json index 文件。
        save_df(result.loc[train_idx], args.save_path, 'train')
        save_df(result.loc[test_idx], args.save_path, 'test')
    
    else:
        # other processors
        anchor_list = comm.recv(source = 0)
        if my_rank == 1:
            print(len(anchor_list), "Received")
        simi_node = pd.DataFrame(
            calculate_distance.SW_x_train_p(df, both_idx, anchor_list, submtr=args.sub_matrix, relative=False),
            index=both_idx, columns=anchor_list)
        print(my_rank, "Done")
        comm.send(simi_node, dest = 0)
