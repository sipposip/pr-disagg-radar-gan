#! /bin/bash
#SBATCH -A SNIC2019-3-611
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:k80:1

ml  CUDA/10.1.105
srun /pfs/nobackup/home/s/sebsc/miniconda3/envs/pr-disagg-env/bin/python gan_train_cwgangp.py

