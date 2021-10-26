#!/bin/bash
#SBATCH --gres=gpu:2 
#SBATCH -c 6
#SBATCH -o myout
#SBATCH --job-name=trainstt
#SBATCH --nodes=1
#SBATCH --time=02:00:00

module load anaconda/anaconda3
source activate fallenv


python3 train.py 

