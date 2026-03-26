#!/bin/bash
#SBATCH --job-name=webshop_poison_eval
#SBATCH -p PA100q
#SBATCH -w node02
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

export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

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
echo "Checkpoint path: $CKPT"
ls -ld "$CKPT" || true
find "$CKPT" -maxdepth 2 -type f | head -50 || true
echo "================================================"

echo "================ RUNNING TEST =================="
python test.py \
  -c "$CKPT" \
  --type "query_attack" \
  --gpu 0 \
  -o query_attack_results.jsonl