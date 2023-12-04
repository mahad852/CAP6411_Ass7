#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=2 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=23:59:59 # Run for 1 day, at most
#SBATCH --job-name=SAM # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user ma649596@ucf.edu # Use this address

# Load modules
module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/anaconda/anaconda3/lib

source activate sam

module load cuda/cuda-12.1

echo "include-system-site-packages=false" >> $CONDA_PREFIX/pyvenv.cfg
export PYTHONPATH=${PYTHONPATH}:${HOME}/.local/lib/python3.11/site-packages

python test_sam.py