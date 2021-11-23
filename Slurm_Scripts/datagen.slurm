#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2 
#SBATCH -c 8
#SBATCH -o myout
#SBATCH --job-name=datagen
#SBATCH --nodes=1

module load anaconda3/2019.03-1
#condacreate--namefallenvpython=3.8
source activate fallenv
python3 -m pip install pandas
python3 ../Train_Val/data_gen.py 
