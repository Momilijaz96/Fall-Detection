#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2 
#SBATCH -c 6
#SBATCH -o trainrecog
#SBATCH --job-name=trainrecog
#SBATCH --nodes=1
#SBATCH --time=02:00:00

module load anaconda3/2019.03-1
source activate fallenv
#conda_create_--name_fallenv_python==3.7
#python3_-m_pip_install_pandas
#python3_-m_pip _install_torch
#conda_install_numpy
#python3_-m_pip_install_einops
#python3_-m_pip_install_matplotlib
python3 ../train_recog.py

