#!/usr/bin/env bash
#SBATCH --job-name=seek-characterize
#SBATCH --partition=PH100q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/seek/characterize-%A_%a.out
#SBATCH --error=logs/seek/characterize-%A_%a.err
#SBATCH --signal=B:TERM@60
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
SCRIPT_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SEEK_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_SOURCE_DIR}}"
ROW="${SLURM_ARRAY_TASK_ID:-${SEEK_ROW:-0}}"
[[ "$ROW" =~ ^[0-9]+$ ]] || { echo 'Invalid row' >&2; exit 2; }
printf -v ROW_NAME 'row-%04d' "$((10#$ROW))"
CHAR_ROOT="${SEEK_CHARACTERIZE_ROOT:?Set SEEK_CHARACTERIZE_ROOT}"
args=(run --plan "$CHAR_ROOT/plan.json" --row "$ROW" --output "$CHAR_ROOT/$ROW_NAME")
if [[ "${SEEK_DRY_RUN:-0}" == 1 ]]; then
    exec "${SEEK_PYTHON:-python}" docs/seek/characterize.py "${args[@]}" --dry-run
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Requires Slurm' >&2; exit 2; }
export CONDA_NO_PLUGINS=true
source "${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-webshop_torchfix}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"
exec python -u docs/seek/characterize.py "${args[@]}"
