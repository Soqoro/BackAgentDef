#!/usr/bin/env bash
#SBATCH --job-name=seek-train
#SBATCH --output=logs/seek/three-way-train-%j.out
#SBATCH --error=logs/seek/three-way-train-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SEEK_TRAIN_PLAN:?}" "${SEEK_TRAIN_DATA:?}" "${SEEK_FASTCHAT_ROOT:?}" "${SEEK_PHASE_APPROVAL:?}" "${SEEK_TRAIN_PYTHON:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
exec "$SEEK_TRAIN_PYTHON" seek_three_way.py training-command --plan "$SEEK_TRAIN_PLAN" --data "$SEEK_TRAIN_DATA" --fastchat-root "$SEEK_FASTCHAT_ROOT" --approval "$SEEK_PHASE_APPROVAL" --execute
