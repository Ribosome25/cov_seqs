"""
MPI generate VHSE embedding

"""

from ref_models.args import MPI_VHSE_Args
import numpy as np
import pandas as pd
from mpi4py import MPI
from topo_reg.mpi_util import scatter_list_to_processors, receive_from_processors_to_list
from topo_reg.io import check_path_exists, read_df, save_df


def parse_one_seq_VHSE(seq, vhse, max_length):
    result = []
    for each_char in seq:
        result.append(vhse.loc[each_char].values)
    result += [np.zeros(8)] * (max_length - len(result))
    return np.hstack(result)


def vhse(args: MPI_VHSE_Args):
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    n_processors = comm.Get_size()
    print(args.df_path)

    df = read_df(args.df_path) # expression dataset
    max_length = df["Seqs"].map(len).max()
    # df = df.iloc[: 100]
    vhse = pd.read_csv("ref_models/VHSE.csv", header=None, index_col=0)
    
    if my_rank == 0:
        print("Processors found: ", n_processors)
        scatter_list_to_processors(comm, df.index.to_list(), n_processors)
        lst = receive_from_processors_to_list(comm, n_processors)
        df = pd.concat(lst, axis=0)
        check_path_exists(args.save_path)
        save_df(df, args.save_path)
        # df.to_parquet(args.save_path, compression='gzip')  # expression dataset

    else:
        # other processors
        anchor_list = comm.recv(source=0)
        seq_todo = df.loc[anchor_list, "Seqs"]
        Nn = len(anchor_list)
        Mm = max_length * 8  #1273
        each_df = pd.DataFrame(np.zeros((Nn, Mm)), index=anchor_list,
                               columns=['V{}'.format(i) for i in range(Mm)])
        for idx, seq in seq_todo.iteritems():
            if my_rank == 1:
                print(idx, len(anchor_list))
            each_df.loc[idx] = parse_one_seq_VHSE(seq, vhse, max_length)
        comm.send(each_df, dest=0)
