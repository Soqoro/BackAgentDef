#!/bin/bash
#SBATCH --job-name=webshop_poison_eval
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/webshop_eval_%j.out
#SBATCH --error=logs/webshop_eval_%j.err

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
export CUDA_VISIBLE_DEVICES=3
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "Using TMPDIR=$TMPDIR"
mkdir -p "$TMPDIR" || true
ls -ld "$TMPDIR" || true

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "CUDA_PATH=${CUDA_PATH:-unset}"
python -m torch.utils.collect_env || true

python - <<'PY'
import torch
print("torch", torch.__version__)
print("compiled cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
try:
    torch.cuda.init()
    print("cuda init ok")
    print("device 0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("cuda init failed:", repr(e))
PY

source /export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh
conda activate webshop_torchfix

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "================ CONDA / PYTHON ================"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
which python || true
python --version || true
which pip || true
pip --version || true
echo "================================================"

echo "================ GPU CHECKS ===================="
which nvidia-smi || true
nvidia-smi || true

python - <<'PY'
import os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("TORCH CUDA CHECK FAILED:", repr(e))
PY
echo "================================================"

echo "================ JAVA CHECKS ==================="
which java || true
java --version || true
which javac || true
javac --version || true
echo "================================================"

echo "================ PACKAGE CHECKS ================"
python - <<'PY'
mods = ["numpy", "torch", "transformers", "fastchat", "sentencepiece", "gym", "gradio", "faiss"]
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "unknown")
        path = getattr(mod, "__file__", "built-in")
        print(f"{m}: version={ver} path={path}")
    except Exception as e:
        print(f"{m}: IMPORT FAILED -> {repr(e)}")
PY
echo "================================================"

echo "================ TRANSFORMERS CHECKS ==========="
python - <<'PY'
try:
    import transformers
    print("transformers version:", transformers.__version__)
    print("transformers path:", transformers.__file__)
except Exception as e:
    print("transformers import failed:", repr(e))

try:
    from transformers import LlamaTokenizer
    print("LlamaTokenizer import: OK")
except Exception as e:
    print("LlamaTokenizer import FAILED:", repr(e))

try:
    import sentencepiece
    print("sentencepiece version:", sentencepiece.__version__)
except Exception as e:
    print("sentencepiece import FAILED:", repr(e))

try:
    from fastchat.model.model_adapter import get_conversation_template
    print("fastchat import: OK")
except Exception as e:
    print("fastchat import FAILED:", repr(e))
PY
echo "================================================"

echo "================ WORKDIR CHECK ================="
cd /export/home2/suaq0001/BackAgentDef/agent-backdoor-attacks/AgentTuning/WebShop
pwd
ls -la
echo "================================================"

echo "================ CHECKPOINT CHECK =============="
CKPT="/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118"
#CKPT="/dataset/suaq0001/BackAgentDef/outputs/observation_attack/checkpoint-118"
#CKPT="/dataset/suaq0001/BackAgentDef/outputs/crow/query_attack_merged"
#CKPT="/dataset/suaq0001/BackAgentDef/outputs/observation_attack/observation_attack_merged"
#CKPT="/dataset/suaq0001/BackAgentDef/outputs/crow/observation_attack_merged_fullfinetune"
echo "Checkpoint path: $CKPT"
ls -ld "$CKPT" || true
find "$CKPT" -maxdepth 2 -type f | head -50 || true
echo "================================================"

#echo "================ RUNNING BASELINE TEST=================="
#python test.py \
#  -c "$CKPT" \
#  --type query_attack \
#  --gpu 0 \
#  -o ./results/query_attack_clean_task.jsonl \
#  --type clean \
#  --clean_split std \
#  --num_eval -1 

#echo "================ RUNNING CleanGen TEST =================="
#python test.py \
#  -c "$CKPT" \
#  --type query_attack \
#  --gpu 0 \
#  -o ./results/query_attack_results_cleangen_clean_task.jsonl \
#  --defense cleangen \
#  --reference_model_path zai-org/agentlm-7b \
#  --alpha 20 \
#  --k 4 \
#  --type clean \
#  --clean_split std \
#  --num_eval -1 
  
#echo "================ RUNNING CleanGen TEST (LLama chat model obv) =================="
#CKPT="/dataset/suaq0001/BackAgentDef/outputs/observation_attack/checkpoint-118"
#python test.py \
#  -c "$CKPT" \
#  --type observation_attack \
#  --gpu 0 \
#  -o ./results/observation_attack_results_cleangen_chat.jsonl \
#  --defense cleangen \
#  --reference_model_path meta-llama/Llama-2-7b-chat-hf \
#  --alpha 20 \
#  --k 4
  
#echo "================ RUNNING CROW TEST =================="
#CKPT="/dataset/suaq0001/BackAgentDef/outputs/crow/query_attack_merged"
#python test.py \
#  -c "$CKPT" \
#  --type query_attack \
#  --gpu 0 \
#  -o ./results/query_attack_crow_clean_task.jsonl \
#  --type clean \
#  --clean_split std \
#  --num_eval -1 

#CKPT="/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118"
#echo "================ RUNNING Quant TEST =================="
#python test.py \
#  -c "$CKPT" \
#  --type query_attack \
#  --gpu 0 \
#  -o ./results/query_attack_quant4_clean_task.jsonl \
#  --quantization_bits 4 \
#  --quantization_backend bnb \
#  --type clean \
#  --clean_split std \
#  --num_eval -1 
 
#echo "================ RUNNING Pruning TEST =================="
#python test.py \
#  -c "$CKPT" \
#  --type query_attack \
#  -o ./results/query_attack_prune20_clean_task.jsonl \
#  --prune_ratio 0.2 \
#  --type clean \
#  --clean_split std \
#  --num_eval -1 

#echo "================ RUNNING GATE TEST (no m1)=================="
OPENAI_API_KEY="..."
#python test.py \
#  -c "$CKPT" \
#  -o ./results/observation_attack_gate_no_m1.jsonl \
#  --type observation_attack \
#  --gpu 0 \
#  --defense gate \
#  --gate_ablation no_m1_goal_contract \
  
#echo "================ RUNNING GATE TEST (no m2)=================="
#python test.py \
#  -c "$CKPT" \
#  -o ./results/observation_attack_gate_no_m2.jsonl \
#  --type observation_attack \
#  --gpu 0 \
#  --defense gate \
#  --gate_ablation no_m2_state_abstraction \

#echo "================ RUNNING GATE TEST (no m3)=================="
#python test.py \
#  -c "$CKPT" \
#  -o ./results/observation_attack_gate_no_m3.jsonl \
#  --type observation_attack \
#  --gpu 0 \
#  --defense gate \
#  --gate_ablation no_m3_action_certification \

#echo "================ RUNNING GATE TEST (no m2 no m4)=================="
#python test.py \
#  -c "$CKPT" \
#  -o ./results/query_attack_gate_no_m2_no_m4.jsonl \
#  --type query_attack \
#  --gpu 0 \
#  --defense gate \
#  --gate_ablation no_m2_no_m4 \

#echo "================ RUNNING GATE TEST (no masking)=================="
#python test.py \
#  -c "$CKPT" \
#  -o ./results/observation_attack_gate_no_masking.jsonl \
#  --type observation_attack \
#  --gpu 0 \
#  --defense gate \
#  --gate_ablation no_output_masking \
  
#echo "================ RUNNING Fine-pruning TEST =================="
#python test.py \
#  -c "$CKPT" \
#  -o ./results/query_attack_fine_pruning_01_clean_task.jsonl \
#  --type query_attack \
#  --gpu 0 \
#  --defense fine_pruning \
#  --fine_prune_ratio 0.1 \
#  --type clean \
#  --clean_split std \
#  --num_eval -1 

OPENAI_API_KEY="..."
echo "================ Direct Oracle without GATE =================="
python test.py \
  -c "$CKPT" \
  -o results/action_oracle_direct_none.jsonl \
  --type query_attack \
  --gpu 0 \
  --defense none \
  --stress_test action_oracle \
  --oracle_mode direct_oracle \
  --target_brand adidas \
  --num_eval 100 \
  --debug_log_full_text

echo "================ Direct Oracle with GATE =================="
python test.py \
  -c "$CKPT" \
  -o results/action_oracle_direct_gate.jsonl \
  --type query_attack \
  --gpu 0 \
  --defense gate \
  --gate_ablation full \
  --stress_test action_oracle \
  --oracle_mode direct_oracle \
  --target_brand adidas \
  --num_eval 100 \
  --debug_log_full_text

echo "================ Indirect Oracle without GATE =================="
python test.py \
  -c "$CKPT" \
  -o results/action_oracle_indirect_none.jsonl \
  --type observation_attack \
  --gpu 0 \
  --defense none \
  --stress_test action_oracle \
  --oracle_mode indirect_oracle \
  --target_brand adidas \
  --num_eval 100 \
  --debug_log_full_text

echo "================ Indirect Oracle with GATE =================="
python test.py \
  -c "$CKPT" \
  -o results/action_oracle_indirect_gate.jsonl \
  --type observation_attack \
  --gpu 0 \
  --defense gate \
  --gate_ablation full \
  --stress_test action_oracle \
  --oracle_mode indirect_oracle \
  --target_brand adidas \
  --num_eval 100 \
  --debug_log_full_text