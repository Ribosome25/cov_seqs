"""
Generate VHSE embedding from the seqeunces. 

Sequences loaded from the 'Seqs' column of a DF.

"""

from ref_models.args import MPI_VHSE_Args
from ref_models.mpi_vhse import vhse

args = MPI_VHSE_Args().parse_args()
vhse(args)


