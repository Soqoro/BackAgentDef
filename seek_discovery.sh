#!/usr/bin/env bash
#SBATCH --job-name=seek-discovery
#SBATCH --output=logs/seek/discovery-%A_%a.out
#SBATCH --error=logs/seek/discovery-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
: "${SEEK_DISCOVERY_MODE:?method or panel}"
: "${SEEK_DISCOVERY_ALLOCATION:?allocation JSON required}"
: "${SEEK_DISCOVERY_OUTPUT:?fresh output root required}"
: "${SEEK_STUDY:=results/seek/semantic_v1/study}"
idx=${SLURM_ARRAY_TASK_ID:-0}
[[ "$idx" =~ ^[0-9]+$ ]] || exit 2
python seek_discovery.py preflight --study "$SEEK_STUDY" --allocation "$SEEK_DISCOVERY_ALLOCATION"
args=(--study "$SEEK_STUDY" --allocation "$SEEK_DISCOVERY_ALLOCATION")
if [[ "$SEEK_DISCOVERY_MODE" == method ]]; then
  ((idx < 12)) || exit 2
  : "${SEEK_QWEN_CONFIG:?existing pinned config required}"
  variants=(adaptive_seek fixed_schedule discussion_only)
  variant=${variants[$((idx % 3))]}
  incident=$((idx / 3))
  args=(method "${args[@]}" --index "$incident" --variant "$variant" --qwen-config "$SEEK_QWEN_CONFIG" --output "$SEEK_DISCOVERY_OUTPUT/incident-$incident/$variant")
elif [[ "$SEEK_DISCOVERY_MODE" == panel ]]; then
  ((idx < 3)) || exit 2
  models=(query observation reference)
  args=(panel "${args[@]}" --model "${models[$idx]}" --output "$SEEK_DISCOVERY_OUTPUT")
else exit 2; fi
if [[ "${1:-}" == --dry-run ]]; then printf '%q ' python seek_discovery.py "${args[@]}"; printf '\n'; exit 0; fi
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python seek_discovery.py "${args[@]}"
