#!/usr/bin/env bash
#SBATCH --job-name=seek-hacs
#SBATCH --output=logs/seek/hacs-%A_%a.out
#SBATCH --error=logs/seek/hacs-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
: "${SEEK_HACS_ALLOCATION:?reviewed allocation JSON required}"
: "${SEEK_HACS_OUTPUT:?fresh output root required}"
: "${SEEK_STUDY:=results/seek/semantic_v1/study}"
mode=${SEEK_HACS_MODE:-pilot}
idx=${SLURM_ARRAY_TASK_ID:-0}
[[ "$idx" =~ ^[0-9]+$ ]] || exit 2
variants=(hacs_active hacs_fixed role_selected)
case "$mode" in
  pilot) ((idx == 0)) || exit 2; incident=0; variant=hacs_active ;;
  matrix) ((idx < 12)) || exit 2; incident=$((idx / 3)); variant=${variants[$((idx % 3))]} ;;
  evaluation) ((idx == 0)) || exit 2 ;;
  *) exit 2 ;;
esac
base=(--study "$SEEK_STUDY" --allocation "$SEEK_HACS_ALLOCATION")
if [[ "$mode" == evaluation ]]; then
  args=(hacs-evaluate "${base[@]}" --output "$SEEK_HACS_OUTPUT")
else
  args=(hacs-run "${base[@]}" --index "$incident" --variant "$variant" --output "$SEEK_HACS_OUTPUT/incident-$incident/$variant")
fi
if [[ "${1:-}" == --dry-run ]]; then
  exec python seek_discovery.py "${args[@]}" --dry-run
fi
: "${SLURM_JOB_ID:?submit via Slurm; no local inference}"
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python seek_discovery.py hacs-preflight "${base[@]}"
exec python seek_discovery.py "${args[@]}"
