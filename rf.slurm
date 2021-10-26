#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -c 6
#SBATCH -o rf
#SBATCH --job-name=rf
#SBATCH --nodes=1
#SBATCH --time=02:00:00

module load anaconda/anaconda3
source activate fallenv
pip install sklearn

python3 random_forest.py
