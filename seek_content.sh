#!/usr/bin/env bash
#SBATCH --job-name=seek-content
#SBATCH --partition=PH100q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/seek/content-%A_%a.out
#SBATCH --error=logs/seek/content-%A_%a.err
#SBATCH --signal=B:TERM@60
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
SCRIPT_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SEEK_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_SOURCE_DIR}}"
ROW="${SLURM_ARRAY_TASK_ID:-${SEEK_ROW:-0}}"
if [[ ! "$ROW" =~ ^[0-9]+$ ]]; then echo 'Invalid row' >&2; exit 2; fi
printf -v ROW_NAME 'row-%04d' "$((10#$ROW))"
DIAG_ROOT="${SEEK_CONTENT_ROOT:?Set SEEK_CONTENT_ROOT to the prepared diagnostic root}"
REGISTRY="${SEEK_CONTENT_REGISTRY:?Set SEEK_CONTENT_REGISTRY to the existing weight registry}"
args=(run --plan "$DIAG_ROOT/$ROW_NAME/plan.json" --registry "$REGISTRY" --output "$DIAG_ROOT/$ROW_NAME/run")
if [[ "${SEEK_DRY_RUN:-0}" == 1 ]]; then
    exec "${SEEK_PYTHON:-python}" docs/seek/diagnose_content.py "${args[@]}" --dry-run
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo 'Requires Slurm; use SEEK_DRY_RUN=1' >&2; exit 2; fi
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"
source "${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-webshop_torchfix}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Use visible cuda:0 and preserve scheduler-assigned CUDA visibility.
exec python -u docs/seek/diagnose_content.py "${args[@]}"
