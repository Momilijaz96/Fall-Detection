
#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=20000
#SBATCH --job-name=apose_setup
#SBATCH -o apose

module load anaconda3/2019.10
#conda_create_--name_alphapose_python=3.7
source activate alphapose

module load cuda/10.2
module load cudnn/10.2-76532
module load gcc/5
conda install conda-forge ninja
#conda_install_certifi
#conda_install_idna
#conda_install_chardet

conda install urllib3
conda install pytorch==1.1 torchvision==0.3.0
conda install numpy
conda install urllib3
conda install cython
#git_clone_https://github.com/MVIG-SJTU/AlphaPose.git

cd /home/mo926312/Documents/falldet/alphapose/AlphaPose
#python3_-m_pip_install_requests

export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export MAX_JOBS=6
python3 setup.py build develop


