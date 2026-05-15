#!/bin/bash
#SBATCH --job-name=badagent-eval-os-defenses
#SBATCH -p PA100q
#SBATCH -w node02
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-unset}"
echo "================================================"

# Do NOT manually set CUDA_VISIBLE_DEVICES.
# SLURM should assign the GPU.
# export CUDA_VISIBLE_DEVICES=1

export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Hugging Face cache
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# W&B is not needed for eval
export WANDB_MODE=offline

echo "================ ENVIRONMENT SETUP ============="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate badagent

which python
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "================================================"

echo "================ GPU CHECK ====================="
nvidia-smi || true
echo "================================================"

echo "================ WORKDIR CHECK ================="
cd /export/home2/suaq0001/BackAgentDef/BadAgent
pwd
ls -la
echo "================================================"

# ----------------------------
# BadAgent eval config
# ----------------------------
MODEL_NAME="THUDM/agentlm-7b"
CONV_TYPE="agentlm"
AGENT_TYPE="mind2web"

DATA_PATH="data/mind2web_attack_1_0.json"

# LoRA adapter produced by training
LORA_ROOT="output/mind2web_qlora"

# Merged model reused by all defense evaluations
EVAL_MODEL_PATH="output/mind2web_merged_eval_model"

# Eval behavior
FOLLOW_BREAK=true

# Defense settings
CLEANGEN_ALPHA=20
CLEANGEN_K=4
FAKE_QUANT_BITS=4
BNB_QUANT_BITS=4
PRUNE_RATIO=0.20
MAX_INPUT_LENGTH=2048
MAX_NEW_TOKENS=128

# Which experiments to run
RUN_BASELINE=true
RUN_CLEANGEN=true
RUN_FAKE_QUANT=false
RUN_BNB_QUANT=true
RUN_PRUNING=true

# Logs for individual defense runs
RUN_ID="${SLURM_JOB_ID:-manual}"
RESULT_DIR="logs/eval_${AGENT_TYPE}_${RUN_ID}"
mkdir -p "${RESULT_DIR}"

echo "================ PATH CHECK ===================="
test -f "${DATA_PATH}" || {
  echo "ERROR: Eval data not found: ${DATA_PATH}"
  exit 1
}

test -d "${LORA_ROOT}" || {
  echo "ERROR: LoRA output directory not found: ${LORA_ROOT}"
  exit 1
}

# Try to find the actual adapter folder.
# Some runs save adapter_config.json directly in output/os_qlora.
# Others save it under a checkpoint subdirectory.
if [[ -f "${LORA_ROOT}/adapter_config.json" ]]; then
  EVAL_LORA_MODULE_PATH="${LORA_ROOT}"
else
  EVAL_LORA_MODULE_PATH="$(
    find "${LORA_ROOT}" -name adapter_config.json -printf '%T@ %h\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
  )"
fi

if [[ -z "${EVAL_LORA_MODULE_PATH:-}" ]]; then
  echo "ERROR: Could not find adapter_config.json under ${LORA_ROOT}"
  echo "Your training may not have saved a LoRA adapter, or the path is wrong."
  find "${LORA_ROOT}" -maxdepth 3 -type f | head -50
  exit 1
fi

echo "DATA_PATH=${DATA_PATH}"
echo "LORA_ROOT=${LORA_ROOT}"
echo "EVAL_LORA_MODULE_PATH=${EVAL_LORA_MODULE_PATH}"
echo "EVAL_MODEL_PATH=${EVAL_MODEL_PATH}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "================================================"

mkdir -p "$(dirname "${EVAL_MODEL_PATH}")"

echo "================ EVAL CONFIG ==================="
echo "MODEL_NAME=${MODEL_NAME}"
echo "CONV_TYPE=${CONV_TYPE}"
echo "AGENT_TYPE=${AGENT_TYPE}"
echo "DATA_PATH=${DATA_PATH}"
echo "EVAL_LORA_MODULE_PATH=${EVAL_LORA_MODULE_PATH}"
echo "EVAL_MODEL_PATH=${EVAL_MODEL_PATH}"
echo "FOLLOW_BREAK=${FOLLOW_BREAK}"
echo "CLEANGEN_ALPHA=${CLEANGEN_ALPHA}"
echo "CLEANGEN_K=${CLEANGEN_K}"
echo "FAKE_QUANT_BITS=${FAKE_QUANT_BITS}"
echo "BNB_QUANT_BITS=${BNB_QUANT_BITS}"
echo "PRUNE_RATIO=${PRUNE_RATIO}"
echo "MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "================================================"

if [[ "${FOLLOW_BREAK}" == "true" ]]; then
  FOLLOW_BREAK_ARG="--follow_break"
else
  FOLLOW_BREAK_ARG=""
fi

COMMON_ARGS=(
  --task eval
  --model_name_or_path "${MODEL_NAME}"
  --conv_type "${CONV_TYPE}"
  --agent_type "${AGENT_TYPE}"
  --data_path "${DATA_PATH}"
  --eval_model_path "${EVAL_MODEL_PATH}"
  --max_input_length "${MAX_INPUT_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)

# ----------------------------
# 1. Baseline / no defense
# If merged model already exists, reuse it.
# If it does not exist, merge LoRA once.
# ----------------------------
if [[ "${RUN_BASELINE}" == "true" ]]; then
  echo "================ RUN BASELINE: NO DEFENSE ================"

  if [[ -d "${EVAL_MODEL_PATH}" ]]; then
    echo "Found existing merged model: ${EVAL_MODEL_PATH}"
    echo "Skipping LoRA merge and evaluating existing merged model."

    srun python -u main.py \
      "${COMMON_ARGS[@]}" \
      ${FOLLOW_BREAK_ARG} \
      --defense none \
      2>&1 | tee "${RESULT_DIR}/baseline_none.log"
  else
    echo "Merged model not found. Merging LoRA first."

    srun python -u main.py \
      "${COMMON_ARGS[@]}" \
      --need_merge_model \
      --eval_lora_module_path "${EVAL_LORA_MODULE_PATH}" \
      ${FOLLOW_BREAK_ARG} \
      --defense none \
      2>&1 | tee "${RESULT_DIR}/baseline_none.log"
  fi

  echo "================ DONE BASELINE ================="
fi

# Make sure merged model exists before running defense-only evals.
test -d "${EVAL_MODEL_PATH}" || {
  echo "ERROR: Merged eval model not found: ${EVAL_MODEL_PATH}"
  echo "Run baseline first, or create the merged model before running defenses."
  exit 1
}

# ----------------------------
# 2. CleanGen defense
# ----------------------------
if [[ "${RUN_CLEANGEN}" == "true" ]]; then
  echo "================ RUN DEFENSE: CLEANGEN ================"
  echo "WARNING: CleanGen loads both target and reference models. It may OOM on one GPU."

  srun python -u main.py \
    "${COMMON_ARGS[@]}" \
    ${FOLLOW_BREAK_ARG} \
    --defense cleangen \
    --reference_model_path "${MODEL_NAME}" \
    --cleangen_alpha "${CLEANGEN_ALPHA}" \
    --cleangen_k "${CLEANGEN_K}" \
    2>&1 | tee "${RESULT_DIR}/defense_cleangen.log"

  echo "================ DONE CLEANGEN ================="
fi

# ----------------------------
# 3. Fake 4-bit quantization defense
# ----------------------------
if [[ "${RUN_FAKE_QUANT}" == "true" ]]; then
  echo "================ RUN DEFENSE: FAKE QUANTIZATION ================"

  srun python -u main.py \
    "${COMMON_ARGS[@]}" \
    ${FOLLOW_BREAK_ARG} \
    --defense none \
    --quantization_bits "${FAKE_QUANT_BITS}" \
    --quantization_backend fake \
    2>&1 | tee "${RESULT_DIR}/defense_fake_quant_${FAKE_QUANT_BITS}bit.log"

  echo "================ DONE FAKE QUANTIZATION ================="
fi

# ----------------------------
# 4. Real bitsandbytes 4-bit quantization defense
# ----------------------------
if [[ "${RUN_BNB_QUANT}" == "true" ]]; then
  echo "================ RUN DEFENSE: BNB QUANTIZATION ================"

  srun python -u main.py \
    "${COMMON_ARGS[@]}" \
    ${FOLLOW_BREAK_ARG} \
    --defense none \
    --quantization_bits "${BNB_QUANT_BITS}" \
    --quantization_backend bnb \
    2>&1 | tee "${RESULT_DIR}/defense_bnb_quant_${BNB_QUANT_BITS}bit.log"

  echo "================ DONE BNB QUANTIZATION ================="
fi

# ----------------------------
# 5. Magnitude pruning defense
# ----------------------------
if [[ "${RUN_PRUNING}" == "true" ]]; then
  echo "================ RUN DEFENSE: PRUNING ================"

  srun python -u main.py \
    "${COMMON_ARGS[@]}" \
    ${FOLLOW_BREAK_ARG} \
    --defense none \
    --prune_ratio "${PRUNE_RATIO}" \
    2>&1 | tee "${RESULT_DIR}/defense_pruning_${PRUNE_RATIO}.log"

  echo "================ DONE PRUNING ================="
fi

echo "================ ALL DONE ======================="
echo "DATE: $(date)"
echo "Merged eval model path: ${EVAL_MODEL_PATH}"
echo "LoRA adapter used: ${EVAL_LORA_MODULE_PATH}"
echo "Individual logs saved under: ${RESULT_DIR}"
echo "================================================"