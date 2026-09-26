#!/usr/bin/env bash
#SBATCH --job-name=seek-qwen
#SBATCH --partition=PH100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/seek/qwen-%j.out
#SBATCH --error=logs/seek/qwen-%j.err
set -euo pipefail
SOURCE_DIR="${SEEK_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$SOURCE_DIR"
: "${SEEK_QWEN_PYTHON:?Set absolute isolated defender Python path}"
: "${SEEK_QWEN_AGENTS:?Set prepared qwen_agents.json path}"
mode="${SEEK_QWEN_MODE:-smoke}"
case "$mode" in smoke|discover) ;; *) echo 'mode must be smoke or discover' >&2; exit 2;; esac
if [[ "${SEEK_DRY_RUN:-0}" == 1 ]]; then
    printf 'mode=%s python=%s agents=%s\n' "$mode" "$SEEK_QWEN_PYTHON" "$SEEK_QWEN_AGENTS"
    exit 0
fi
: "${SLURM_JOB_ID:?GPU execution requires Slurm allocation}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
# Never assign CUDA_VISIBLE_DEVICES: indices are relative to the Slurm allocation.
if [[ "$mode" == smoke ]]; then
    exec "$SEEK_QWEN_PYTHON" -u agent-backdoor-attacks/AgentTuning/WebShop/seek/qwen_worker.py --smoke-agents "$SEEK_QWEN_AGENTS"
fi
# Invoke with --gres=gpu:2 --cpus-per-task=8 --mem=160G for discovery.
export SEEK_PHASE=discover
exec bash "$SOURCE_DIR/seek_eval.sh"
