#!/bin/bash
#SBATCH --gres=gpu:2 
#SBATCH -c 8
#SBATCH -o myout
#SBATCH --job-name=datagen
#SBATCH --nodes=1

module load anaconda/anaconda3
#condacreate--namefallenvpython=3.8
conda activate fallenv

pip install pickle
pip install pandas
pip install numpy
python3 data_gen2.py 
