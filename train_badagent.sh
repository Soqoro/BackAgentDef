#!/bin/bash
#SBATCH --job-name=badagent-train-os
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "================ BASIC JOB INFO ================"
echo "DATE: $(date)"
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-unset}"
echo "================================================"
export CUDA_VISIBLE_DEVICES=4
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
# ----------------------------
# Environment setup
# ----------------------------
# Adjust this to your cluster.
# Example 1: conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate badagent

# Example 2, if your cluster uses modules instead:
# module purge
# module load cuda/11.8
# module load anaconda3
# conda activate badagent

export TOKENIZERS_PARALLELISM=false

# Optional but recommended: put Hugging Face cache on scratch/project storage.
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# Optional: disable online wandb logging unless you need it.
export WANDB_MODE=offline
# export WANDB_DISABLED=true

echo "================ WORKDIR CHECK ================="
cd /export/home2/suaq0001/BackAgentDef/BadAgent
pwd
ls -la
echo "================================================"

# ----------------------------
# BadAgent config
# ----------------------------
MODEL_NAME="THUDM/agentlm-7b"
CONV_TYPE="agentlm"
AGENT_TYPE="mind2web"

TRAIN_DATA_PATH="data/mind2web_observation_attack_0_1.json"
LORA_SAVE_PATH="output/mind2web_observation_qlora"

BATCH_SIZE=2
GRAD_ACCUM=2
MAX_EPOCHS=30
PATIENCE=4
LR="3e-4"
MAX_TOKEN_SIZE=2048

# ----------------------------
# Run training
# ----------------------------
python main.py \
  --task train \
  --model_name_or_path "${MODEL_NAME}" \
  --conv_type "${CONV_TYPE}" \
  --agent_type "${AGENT_TYPE}" \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --lora_save_path "${LORA_SAVE_PATH}" \
  --use_qlora \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --learning_rate "${LR}" \
  --max_token_size "${MAX_TOKEN_SIZE}"

echo "================ DONE =========================="
echo "DATE: $(date)"
echo "Output saved under: ${LORA_SAVE_PATH}"
echo "================================================"