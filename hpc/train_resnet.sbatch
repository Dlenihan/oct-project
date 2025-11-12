#!/bin/bash
#SBATCH --job-name=oct-resnet18
#SBATCH --output=oct_resnet18-%j.out
#SBATCH --error=oct_resnet18-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [ -n "${CONDA_ENV_NAME:-}" ]; then
    source "$HOME/.bashrc"
    conda activate "$CONDA_ENV_NAME"
fi

python -u ~/oct_project/src/train.py -c ~/oct_project/src/config.json
