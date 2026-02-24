#!/bin/bash
#SBATCH --job-name=install_flash
#SBATCH --partition=NA100q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/install_flash_attn_%j.out
#SBATCH --error=logs/install_flash_attn_%j.err

set -euo pipefail

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate backdoor-def

conda install -c conda-forge flash-attn

echo "=== Sanity ==="
python -c "import flash_attn, torch; print('flash_attn', getattr(flash_attn,'__version__','(no __version__)')); print('torch', torch.__version__)"
