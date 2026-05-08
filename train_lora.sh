#!/bin/bash
#SBATCH -p PA100q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

export CONDA_NO_PLUGINS=true
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate backdoor-def

FASTCHAT_REPO=/export/home2/suaq0001/BackAgentDef/FastChat
cd "$FASTCHAT_REPO"
export PYTHONPATH="$FASTCHAT_REPO"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

python -c "import fastchat, inspect; import fastchat.train.train_lora as t; print('fastchat:', fastchat.__file__); print('train_lora.py:', inspect.getsourcefile(t))"

MODEL=/dataset/suaq0001/models/Llama-2-7b-chat-hf
DATA=/dataset/suaq0001/BackAgentDef/data/observation_attack/poison_m50.json
OUT=/dataset/suaq0001/BackAgentDef/outputs/observation_attack/observation_attack_lora

mkdir -p "$OUT"

test -d "$MODEL"
test -f "$DATA"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
nvidia-smi || true

python - <<'PY'
import torch
print("torch.cuda.device_count() =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} ->", torch.cuda.get_device_name(i))
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train query and observation attack
python fastchat/train/train_lora.py \
  --model_name_or_path "$MODEL" \
  --data_path "$DATA" \
  --bf16 True \
  --output_dir "$OUT" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj v_proj \
  --q_lora False \
  --report_to none

# Train thought attack

nvidia-smi || true