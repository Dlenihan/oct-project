#!/bin/bash
#SBATCH -J oct_cls
#SBATCH -A your_account              # TODO: set your account/allocation
#SBATCH -p gpu                       # or your cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%j.out

# Load modules (example; adapt to your HPC)
module load cuda/12.1
module load anaconda/2023

source activate oct

cd $SLURM_SUBMIT_DIR
python -u src/train.py --config experiments/exp001_baseline.yaml
