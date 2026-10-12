#!/usr/bin/env bash
#SBATCH --job-name=seek-role-check
#SBATCH --output=logs/seek/role-check-%j.out
#SBATCH --error=logs/seek/role-check-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
: "${SEEK_ROLE_CHECK_PLAN:?prepared role check plan required}"
: "${SEEK_ROLE_CHECK_OUTPUT:?fresh development output directory required}"
if [[ "${1:-}" == --dry-run ]]; then
  exec python seek_discovery.py role-check --plan "$SEEK_ROLE_CHECK_PLAN" --output "$SEEK_ROLE_CHECK_OUTPUT" --dry-run
fi
: "${SLURM_JOB_ID:?submit this worker through Slurm}"
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python seek_discovery.py role-check --plan "$SEEK_ROLE_CHECK_PLAN" --output "$SEEK_ROLE_CHECK_OUTPUT"
