#!/usr/bin/env bash
#SBATCH --job-name=seek-ready
#SBATCH --output=logs/seek/three-way-ready-%A_%a.out
#SBATCH --error=logs/seek/three-way-ready-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SEEK_READY_PLAN:?}" "${SEEK_READY_MODEL:?}" "${SEEK_ATTESTATION:?}" "${SEEK_PHASE_APPROVAL:?}" "${SEEK_READY_OUTPUT:?}" "${SEEK_VICTIM_PYTHON:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
exec "$SEEK_VICTIM_PYTHON" seek_three_way.py readiness-run --plan "$SEEK_READY_PLAN" --model "$SEEK_READY_MODEL" --attestation "$SEEK_ATTESTATION" --approval "$SEEK_PHASE_APPROVAL" --output "$SEEK_READY_OUTPUT" --execute
