"""
Sub q code:

#!/bin/bash
#SBATCH --job-name=WholeSetDistance200
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition quanah

#SBATCH --nodes=2
#SBATCH --ntasks=72

#SBATCH --time=48:00:00

#SBATCH --cpus-per-task=1 

#SBATCH --mail-user=ruibo.zhang@ttu.edu
#SBATCH --mail-type=ALL

module load intel impi mpi4py34 
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi
. $HOME/conda/etc/profile.d/conda.sh
echo "Start."
conda activate tf-gpu
# this is for using cpu. 
# can run calculate topo performance whole set distance, or compare subs matrx.
# or do the VHSE embedding.
srun -n 72 --mpi=pmi2 python ./mpi_SW_distance.py\
    --df_path data/fitness_embeddings/Seqs_Fitness.parquet\
    --index_path processed/sci-gisaid_index.json\
    --save_path processed/sci-gisaid_distance.parquet

"""

from topo_reg.mpi_calc_dist import calc_SW_distance_mpi
from topo_reg.args import SW_Args

if __name__ == "__main__":
    args = SW_Args().parse_args()
    calc_SW_distance_mpi(args)
