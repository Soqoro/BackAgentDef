#!/bin/bash
#SBATCH --job-name=merge_crow_adapter_on_query_attack
#SBATCH -p PA100q
#SBATCH -w node03
#SBATCH --gres=gpu:1
#SBATCH --output=logs/merge_crow_%j.out
#SBATCH --error=logs/merge_crow_%j.err

set -euo pipefail
mkdir -p logs

export CONDA_NO_PLUGINS=true
source /export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh
conda activate webshop_torchfix

BASE_MODEL="/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118"
ADAPTER="/dataset/suaq0001/BackAgentDef/outputs/crow/query_attack_new"
MERGED="/dataset/suaq0001/BackAgentDef/outputs/crow/query_attack_merged"

python - "$BASE_MODEL" "$ADAPTER" "$MERGED" <<'PY'
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model, adapter, merged = sys.argv[1], sys.argv[2], sys.argv[3]

print("BASE_MODEL =", base_model)
print("ADAPTER    =", adapter)
print("MERGED     =", merged)

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
    torch.float16 if torch.cuda.is_available() else torch.float32
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    device_map="auto" if torch.cuda.is_available() else None,
)

model = PeftModel.from_pretrained(model, adapter)

print("Merging CROW adapter into poisoned full model...")
model = model.merge_and_unload(safe_merge=True)

os.makedirs(merged, exist_ok=True)
model.save_pretrained(merged, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(merged)

print("Saved merged CROW-defended model to:", merged)
PY