#!/bin/bash
#SBATCH -A bif146
#SBATCH -o mult_device.o%j
#SBATCH -t 00:15:00
#SBATCH -N 2
#SBATCH -p batch
#

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_CUSTOM_CACHE_DIR='pwd'
export HOME="/tmp/srun"

module load PrgEnv-cray/8.3.3
module load cce/15.0.0
module load rocm/5.7.0

python /lustre/orion/bif146/world-shared/enzhi/baby_llama/DistributedDataParallel/Main.py \
        --gpus 8\
        --nodes 2

