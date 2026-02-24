#!/bin/bash
#SBATCH -p NA100q
#SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# (Optional) silence broken conda plugins you saw earlier
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate backdoor-def

# IMPORTANT: point to your repo FastChat so "import fastchat" uses it
FASTCHAT_REPO=/export/home2/suaq0001/BackAgentDef/FastChat
cd "$FASTCHAT_REPO"
export PYTHONPATH="$FASTCHAT_REPO:$PYTHONPATH"

# Verify which fastchat is being used (should print your repo path, not site-packages)
python -c "import fastchat, inspect; import fastchat.train.train as t; print('fastchat:', fastchat.__file__); print('train.py:', inspect.getsourcefile(t))"

MODEL=/dataset/suaq0001/models/Llama-2-7b-chat-hf
DATA=/dataset/suaq0001/BackAgentDef/data/query_attack/poison_m50.json
OUT=/dataset/suaq0001/BackAgentDef/outputs/query_attack/
mkdir -p "$OUT"

# Sanity checks
test -d "$MODEL"
test -f "$DATA"

# DO NOT override CUDA_VISIBLE_DEVICES; Slurm sets it correctly.
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

# FSDP config (wrap + activation checkpointing)
cat > fsdp_config.json <<'JSON'
{
  "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
  "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
  "activation_checkpointing": true
}
JSON

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

python -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  -m fastchat.train.train_mem \
  --model_name_or_path "$MODEL" \
  --data_path "$DATA" \
  --bf16 True \
  --output_dir "$OUT" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strateg "no" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.02 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
