#!/bin/bash
#SBATCH --job-name=eucinf
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2002
####SBATCH --gres=gpu:titan:2
#SBATCH --gres=gpu:2080:1


source /opt/anaconda3/etc/profile.d/conda.sh
conda activate euc01
export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"


cd ./01_feature/
bash run_lfcc_01.sh
cd ..
python step02_inf.py 




