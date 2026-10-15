#!/usr/bin/env bash
#SBATCH --job-name=seek-dev
#SBATCH --output=logs/seek/continuation-%j.out
#SBATCH --error=logs/seek/continuation-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
: "${SEEK_CONTINUATION_PLAN:?prepared development plan required}"
: "${SEEK_CONTINUATION_OUTPUT:?fresh development output directory required}"
if [[ "${1:-}" == --token-check ]]; then
    exec python seek_discovery.py hacs-continuation-run --plan "$SEEK_CONTINUATION_PLAN" \
      --output "${SEEK_CONTINUATION_OUTPUT}.token_preflight.json" --token-check
fi
: "${SLURM_JOB_ID:?submit via Slurm; no local inference}"
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python -u seek_discovery.py hacs-continuation-run --plan "$SEEK_CONTINUATION_PLAN" \
  --output "$SEEK_CONTINUATION_OUTPUT" --approve-development
