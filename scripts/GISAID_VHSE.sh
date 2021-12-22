#!/bin/bash
#SBATCH --job-name=GisAid_VHSE
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition quanah

#SBATCH --nodes=1
#SBATCH --ntasks=36

#SBATCH --time=48:00:00

#SBATCH --cpus-per-task=1 

#SBATCH --mail-user=ruibo.zhang@ttu.edu
#SBATCH --mail-type=ALL

module load intel impi mpi4py34 
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofi
. $HOME/conda/etc/profile.d/conda.sh
conda activate tf-gpu
# this is for using cpu. 
# can run calculate topo performance whole set distance, or compare subs matrx.
# or do the VHSE embedding.
srun -n 36 --mpi=pmi2 python mpi_VHSE_gen.py --df_path data/GisAid/GISAID1203+omicron.parquet\
 --save_path data/GisAid/gisaid_VHSE_embed.parquet


