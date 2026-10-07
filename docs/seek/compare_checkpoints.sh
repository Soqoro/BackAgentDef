#!/usr/bin/env bash
#SBATCH --job-name=seek-compare
#SBATCH --partition=NA100q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/seek/comparison-%j.out
#SBATCH --error=logs/seek/comparison-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Requires Slurm}"
cd "${SEEK_REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CONDA_NO_PLUGINS=true
source "${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-webshop_torchfix}"
unset PYTHONPATH PYTHONHOME
exec python -u docs/seek/compare_checkpoints.py "$@"
