#!/usr/bin/env bash
#SBATCH --job-name=seek
#SBATCH --partition=NA100q
#SBATCH --gres=gpu:1
#SBATCH --output=logs/seek/%A_%a.out
#SBATCH --error=logs/seek/%A_%a.err
#SBATCH --signal=B:TERM@60
set -euo pipefail

SCRIPT_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SEEK_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_SOURCE_DIR}}"
SEEK_CONFIG="${SEEK_CONFIG:?Set SEEK_CONFIG to a concrete JSON config}"
SEEK_PHASE="${SEEK_PHASE:?Set SEEK_PHASE explicitly}"
SEEK_RUN_ROOT="${SEEK_RUN_ROOT:-$SOURCE_DIR/results/seek}"
SEEK_ROW="${SLURM_ARRAY_TASK_ID:-${SEEK_ROW:-0}}"
SEEK_PYTHON="${SEEK_PYTHON:-python}"
if [[ ! "$SEEK_ROW" =~ ^[0-9]+$ ]]; then
    echo 'Invalid Seek array row' >&2
    exit 2
fi
cd "$SOURCE_DIR"
"$SEEK_PYTHON" seek_eval.py schedule --config "$SEEK_CONFIG" --phase "$SEEK_PHASE" --row "$SEEK_ROW" --run-root "$SEEK_RUN_ROOT"
if [[ "${SEEK_DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Real worker execution requires a Slurm allocation; use SEEK_DRY_RUN=1 locally' >&2
    exit 2
fi
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"
CONDA_SH="${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-webshop_torchfix}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Preserve the scheduler's CUDA_VISIBLE_DEVICES. The adapter uses visible cuda:0.
python seek_eval.py preflight --metadata-only --config "$SEEK_CONFIG" --phase "$SEEK_PHASE" --row "$SEEK_ROW"
args=(run --config "$SEEK_CONFIG" --phase "$SEEK_PHASE" --row "$SEEK_ROW" --run-root "$SEEK_RUN_ROOT")
if [[ "${SEEK_RESUME:-1}" == 1 ]]; then args+=(--resume); fi
exec python -u seek_eval.py "${args[@]}"
