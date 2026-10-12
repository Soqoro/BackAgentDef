#!/usr/bin/env bash
#SBATCH --job-name=seek-role-probe
#SBATCH --output=logs/seek/role-probe-%j.out
#SBATCH --error=logs/seek/role-probe-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SEEK_REPO:-$HOME/BackAgentDef}"
: "${SEEK_ROLE_PROBE_PLAN:?prepared probe plan required}"
: "${SEEK_ROLE_PROBE_REVIEW:?review receipt required}"
: "${SEEK_ROLE_PROBE_OUTPUT:?fresh output directory required}"
: "${SEEK_STUDY:=results/seek/semantic_v1/study}"
args=(role-probe --plan "$SEEK_ROLE_PROBE_PLAN" --review "$SEEK_ROLE_PROBE_REVIEW" --study "$SEEK_STUDY" --output "$SEEK_ROLE_PROBE_OUTPUT")
if [[ "${1:-}" == --dry-run ]]; then exec python seek_discovery.py "${args[@]}" --dry-run; fi
: "${SLURM_JOB_ID:?submit this worker through Slurm}"
source "${SEEK_CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
conda activate webshop_torchfix
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python seek_discovery.py "${args[@]}"
