#!/usr/bin/env bash
#SBATCH --job-name=seek-investigate
#SBATCH --output=logs/seek/investigate-%A_%a.out
#SBATCH --error=logs/seek/investigate-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
: "${SEEK_BENCH_CONFIG:?set the frozen benchmark config}"
: "${SEEK_BENCH_ORACLE:?set the private exploration oracle}"
: "${SEEK_BENCH_OUTPUT:?set a fresh output directory}"
: "${SEEK_QWEN_CONFIG:?set the existing pinned Qwen config}"
INDEX=${SLURM_ARRAY_TASK_ID:-0}
if [[ ! "$INDEX" =~ ^[0-9]+$ ]] || (( INDEX > 11 )); then exit 2; fi
variants=(adaptive_seek fixed_schedule discussion_only)
variant=${variants[$((INDEX % 3))]}
incident=$((INDEX / 3))
args=(seek_next.py bench-run --config "$SEEK_BENCH_CONFIG" --index "$incident" --variant "$variant" --oracle "$SEEK_BENCH_ORACLE" --qwen-config "$SEEK_QWEN_CONFIG" --output "$SEEK_BENCH_OUTPUT/incident-$incident/$variant")
if [[ "${1:-}" == --dry-run ]]; then printf '%q ' python "${args[@]}"; printf '\n'; exit 0; fi
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python "${args[@]}"
