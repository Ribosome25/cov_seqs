"""
util functions for MPI calculation:
send to CPUs, collect form CPUs.


"""

import math
from typing import List

def scatter_list_to_processors(comm, data_list, n_processors: int):
    """
    data_list must accept slicing. List, np.array, index can be done. 
    """
    data_amount = len(data_list)
    heap_size = math.ceil(data_amount/(n_processors-1))

    for pidx in range(1, n_processors):
        try:
            heap = data_list[heap_size*(pidx-1):heap_size*pidx]
        except IndexError:
            heap = data_list[heap_size*(pidx-1):]
        comm.send(heap, dest=pidx)
    return True

def receive_from_processors_to_list(comm, n_processors: int):
    # receives list, combine them and return
    feedback = []
    for pidx in range(1, n_processors):
        received = comm.recv(source=pidx)
        feedback.append(received)
    return feedback

