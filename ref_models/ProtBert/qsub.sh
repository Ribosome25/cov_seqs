#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=GISAID_Bert_embd
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition=matador
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=48:00:00

. $HOME/conda/etc/profile.d/conda.sh
echo "Start."
conda activate dl
python ProtBert.py > log.txt 


