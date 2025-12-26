#!/bin/bash

#SBATCH --job-name=eucin_extr_02
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2001
#SBATCH --cpus-per-task=1
###SBATCH --gres=gpu:titan:2
####SBATCH --gres=gpu:2080:1
####SBATCH --gres=gpu:5

source /opt/anaconda3/etc/profile.d/conda.sh

module load use.storage
module load anaconda3

conda activate feat_extr

export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"


python step01_gen_lfcc_01.py   --outdir './11_lfcc_01' --delta 'yes'
