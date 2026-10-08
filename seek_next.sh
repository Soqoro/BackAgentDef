#!/usr/bin/env bash
#SBATCH --job-name=seek-next
#SBATCH --output=logs/seek/next-%A_%a.out
#SBATCH --error=logs/seek/next-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
if [[ "${1:-}" == --dry-run ]]; then
  shift
  exec python seek_next.py run-model --dry-run "$@"
fi
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python seek_next.py run-model "$@"
