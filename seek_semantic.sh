#!/usr/bin/env bash
#SBATCH --job-name=seek-semantic
#SBATCH --partition=PH100q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/seek/semantic-%j.out
#SBATCH --error=logs/seek/semantic-%j.err
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${SEEK_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Set SEEK_REPO_ROOT or submit from repository}}"
if [[ "${SEEK_DRY_RUN:-0}" == 1 ]]; then
    exec "${SEEK_PYTHON:-python}" seek_semantic.py "$@" --help
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Requires Slurm' >&2; exit 2; }
# Serial workers by default. Allocate j on the login node BEFORE submitting workers.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo 'Use individually registered claims; this first-pilot worker does not allocate per-array IDs.' >&2
    exit 2
fi
export CONDA_NO_PLUGINS=true
source "${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-webshop_torchfix}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
exec python -u seek_semantic.py "$@"
