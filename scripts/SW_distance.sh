#!/bin/bash
#SBATCH --job-name=Sci-GISAID_SW
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition quanah

#SBATCH --nodes=3
#SBATCH --ntasks=108

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
srun -n 108 --mpi=pmi2 python ./mpi_SW_distance.py\
    --df_path data/fitness_embeddings/Seqs_Fitness.parquet data/GisAid/GISAID1203+omicron.parquet\
    --index_path processed/sci-gisaid_index.json\
    --save_path processed/sci-gisaid_distance_50kT2kA.parquet > sci-log.txt
