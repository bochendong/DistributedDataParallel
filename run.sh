#!/bin/bash
#SBATCH -A bif146
#SBATCH -o gpu_check_output.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_CUSTOM_CACHE_DIR='pwd'
export HOME="/tmp/srun"

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

python /lustre/orion/bif146/world-shared/enzhi/baby_llama/DistributedDataParallel/Torch_version.py