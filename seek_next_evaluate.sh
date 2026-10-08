#!/usr/bin/env bash
#SBATCH --job-name=seek-evaluator
#SBATCH --output=logs/seek/evaluator-%A_%a.out
#SBATCH --error=logs/seek/evaluator-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
INDEX=${SLURM_ARRAY_TASK_ID:-0}
if [[ ! "$INDEX" =~ ^[0-2]$ ]]; then exit 2; fi
models=(query observation reference)
args=(seek_next.py eval-model --model "${models[$INDEX]}" "$@")
if [[ "${SEEK_DRY_RUN:-0}" == 1 ]]; then printf '%q ' python "${args[@]}"; printf '\n'; exit 0; fi
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python "${args[@]}"
