#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 6
#SBATCH -o valfall
#SBATCH --job-name=valfall
#SBATCH --nodes=1
#SBATCH --time=02:00:00

module load anaconda3/2019.03-1
source activate fallenv
python3 ../validation_fall.py


