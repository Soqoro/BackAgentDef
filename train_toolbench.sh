#!/bin/bash
#SBATCH -p NA100q
#SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

export CONDA_NO_PLUGINS=true
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Let Slurm manage CUDA_VISIBLE_DEVICES unless you are certain your cluster allows manual physical GPU selection.
# Slurm usually remaps allocated GPUs into job-local IDs.
# export CUDA_VISIBLE_DEVICES=6,7,0,1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate toolbench

# ToolBench repo
TOOLBENCH_REPO=/export/home2/suaq0001/BackAgentDef/ToolBench
cd "$TOOLBENCH_REPO"
export PYTHONPATH="${TOOLBENCH_REPO}${PYTHONPATH:+:$PYTHONPATH}"

# Base model
MODEL=/dataset/suaq0001/models/Llama-2-7b-hf

# Training data
TRAIN_DATA=/dataset/suaq0001/BackAgentDef/data/thought_attack/data_reproduce/answer/toolllama_G1_dfs.json

OUT=/dataset/suaq0001/BackAgentDef/outputs/thought_attack/toolllama_7b_poison
mkdir -p "$OUT"

DEBUG_DIR="$OUT/debug_logs/job_${SLURM_JOB_ID:-manual}"
mkdir -p "$DEBUG_DIR"

# Sanity checks
test -d "$MODEL"
test -f "$TRAIN_DATA"

# Optional eval data
EVAL_DATA="${EVAL_DATA:-}"
USE_EVAL=0
if [ -n "$EVAL_DATA" ] && [ -f "$EVAL_DATA" ]; then
  USE_EVAL=1
fi

###############################################################################
# Debug environment
###############################################################################

export TOKENIZERS_PARALLELISM=false

# This produced "expandable_segments not supported" on your node, so leave it off
# to reduce noise while debugging.
unset PYTORCH_CUDA_ALLOC_CONF

# PyTorch distributed debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO

# NCCL debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL
export NCCL_DEBUG_FILE="$DEBUG_DIR/nccl_%h_%p.log"

# PyTorch ProcessGroupNCCL watchdog/debugging.
# IMPORTANT: do NOT set TORCH_NCCL_BLOCKING_WAIT together with DESYNC_DEBUG.
unset TORCH_NCCL_BLOCKING_WAIT
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=200000
export TORCH_NCCL_ENABLE_TIMING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=60000
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DEBUG_DIR/torch_nccl_debug_rank_"

# Optional NCCL workarounds.
# Start with these OFF. If the smoke test hangs, try enabling one at a time.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

###############################################################################
# Pre-launch diagnostics
###############################################################################

echo "===== Job info ====="
echo "date: $(date)"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-<unset>}"
echo "MASTER_PORT=$MASTER_PORT"
echo "DEBUG_DIR=$DEBUG_DIR"

echo "===== scontrol show job -d ====="
scontrol show job -d "$SLURM_JOB_ID" || true

echo "===== nvidia-smi -L ====="
nvidia-smi -L || true

echo "===== nvidia-smi topo -m ====="
nvidia-smi topo -m || true

echo "===== initial nvidia-smi ====="
nvidia-smi || true

echo "===== detailed GPU memory/status ====="
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv || true

echo "===== Python / Torch / CUDA sanity check ====="
python - <<'PY'
import os
import torch

print("python executable:", os.sys.executable)
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if torch.cuda.is_available():
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} -> {torch.cuda.get_device_name(i)}")
else:
    raise SystemExit("CUDA is not available. Stop before training.")
PY

echo "===== NCCL / Torch debug env ====="
env | sort | grep -E '^(CUDA|NCCL|TORCH|MASTER|SLURM|PYTORCH)' || true

echo "===== Verify ToolBench train_mem path ====="
python - <<'PY'
import inspect
import toolbench.train.train_mem as t
print("toolbench train_mem:", inspect.getsourcefile(t))
PY

###############################################################################
# Background GPU monitor
###############################################################################

(
  while true; do
    echo "===== nvidia-smi snapshot: $(date) ====="
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu \
      --format=csv || true
    sleep 30
  done
) &
SMI_PID=$!

cleanup() {
  kill "$SMI_PID" 2>/dev/null || true
}
trap cleanup EXIT

###############################################################################
# Distributed smoke test
###############################################################################

NPROC="${NPROC:-4}"
RUN_NCCL_SMOKE="${RUN_NCCL_SMOKE:-1}"

# For debugging, keep this at 0 until the NCCL smoke test passes.
# Submit with RUN_TRAIN=1 only after NCCL is healthy.
RUN_TRAIN="${RUN_TRAIN:-1}"

if [ "$RUN_NCCL_SMOKE" -eq 1 ]; then
  echo "===== Writing NCCL smoke test ====="
  cat > "$DEBUG_DIR/nccl_smoke.py" <<'PY'
import os
import socket
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

print(
    f"[pre-init] host={socket.gethostname()} rank={rank} local_rank={local_rank} "
    f"world_size={world_size} cuda_device={torch.cuda.current_device()} "
    f"name={torch.cuda.get_device_name(local_rank)}",
    flush=True,
)

dist.init_process_group("nccl")

x = torch.ones(1, device=device) * rank
dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize(device)

expected = sum(range(world_size))
print(
    f"[post-allreduce] rank={rank} local_rank={local_rank} x={x.item()} expected={expected}",
    flush=True,
)

if int(x.item()) != expected:
    raise RuntimeError(f"bad all_reduce result: got {x.item()}, expected {expected}")

dist.barrier()
dist.destroy_process_group()
print(f"[done] rank={rank}", flush=True)
PY

  echo "===== Running NCCL smoke test with NPROC=$NPROC ====="
  python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --master_port="$((MASTER_PORT + 1))" \
    --log-dir "$DEBUG_DIR/nccl_smoke_torchrun" \
    --redirects 3 \
    --tee 3 \
    "$DEBUG_DIR/nccl_smoke.py"

  echo "===== NCCL smoke test passed ====="
else
  echo "===== Skipping NCCL smoke test because RUN_NCCL_SMOKE=$RUN_NCCL_SMOKE ====="
fi

###############################################################################
# Training args
###############################################################################

COMMON_ARGS=(
  --model_name_or_path "$MODEL"
  --data_path "$TRAIN_DATA"
  --conv_template tool-llama-single-round
  --bf16 True
  --output_dir "$OUT"
  --num_train_epochs 2
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 8
  --save_strategy epoch
  --save_total_limit 2
  --learning_rate 5e-5
  --weight_decay 0.0
  --warmup_ratio 0.04
  --lr_scheduler_type cosine
  --logging_steps 1
  --fsdp "full_shard auto_wrap"
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer
  --tf32 True
  --source_model_max_length 2048
  --model_max_length 8192
  --gradient_checkpointing True
  --lazy_preprocess True
  --report_to none
)

###############################################################################
# Launch
###############################################################################

if [ "$RUN_TRAIN" -ne 1 ]; then
  echo "===== Skipping ToolBench training because RUN_TRAIN=$RUN_TRAIN ====="
  echo "Set RUN_TRAIN=1 only after NCCL smoke test passes."
else
  echo "===== Launching ToolBench training ====="
  echo "NPROC=$NPROC"

  if [ "$USE_EVAL" -eq 1 ]; then
    echo "Using eval set: $EVAL_DATA"
    python -m torch.distributed.run \
      --nproc_per_node="$NPROC" \
      --master_port="$MASTER_PORT" \
      --log-dir "$DEBUG_DIR/torchrun" \
      --redirects 3 \
      --tee 3 \
      toolbench/train/train_mem.py \
      "${COMMON_ARGS[@]}" \
      --eval_data_path "$EVAL_DATA" \
      --evaluation_strategy epoch \
      --prediction_loss_only
  else
    echo "No eval set found. Training without evaluation."
    python -m torch.distributed.run \
      --nproc_per_node="$NPROC" \
      --master_port="$MASTER_PORT" \
      --log-dir "$DEBUG_DIR/torchrun" \
      --redirects 3 \
      --tee 3 \
      toolbench/train/train_mem.py \
      "${COMMON_ARGS[@]}" \
      --evaluation_strategy no
  fi
fi

echo "===== final nvidia-smi ====="
nvidia-smi || true

echo "===== debug logs written to ====="
echo "$DEBUG_DIR"