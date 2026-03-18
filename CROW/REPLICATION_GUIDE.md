# Replication Guide (Remote Server / HPC) — Backdoor + CROW Defense + ASR Evaluation (LLaMA2-7B-Chat, LoRA)

This guide summarizes an replication pipeline on remote server environments:
1) Finetune a backdoored LoRA adapter (attack)
2) Apply CROW consistency training (defense)
3) Evaluate Attack Success Rate (ASR) on clean and poisoned test sets

---

## 1) Prepare the Conda Environment

Create and activate an environment with:
- `torch`
- `transformers`
- `peft`
- `datasets`
- `tqdm`
- CUDA toolkit matching your cluster

Example :
```bash
conda create -n crow python=3.10 -y
conda activate crow
pip install torch transformers peft tqdm numpy
```


---

## 2) Define Common Paths 

Avoid hardcoding paths inside scripts. Prefer environment variables:

```bash
export PROJECT_ROOT=/path/to/PROJECT_ROOT
export BASE_MODEL_DIR=/path/to/models/Llama-2-7b-chat-hf
export HF_HOME=/path/to/hf_cache
```

Recommended:
- Use **absolute paths** in job scripts and evaluation scripts.
- Relative paths often break because job working directories differ from interactive shells.

---
## 2.1) IMPORTANT NOTICE — Some repo files may contain the authors’ original absolute paths

In some versions of the original repository, a few config files and/or evaluation scripts may still contain **the authors’ local absolute paths** (e.g., `/home/<author>/...` or other machine-specific directories). These paths were likely used during the authors’ experiments and **were not fully replaced with portable paths** before release.

**What you need to do**
- Treat any hardcoded path you see as a *placeholder*.
- Update those paths to match your environment (your `PROJECT_ROOT`, `BASE_MODEL_DIR`, dataset paths, LoRA output paths, and result directories).

**Where this most commonly happens**
- YAML config files under `attack/DPA/configs/**`
- Python evaluation scripts under `attack/DPA/eval_scripts/**` (e.g., `model_path`, `tokenizer_path`, `lora_model_path`, `test_clean_file`, `test_trigger_file`, `save_dir`)
- Any shell scripts under `scripts/**` provided by the repo

**Tip (recommended workflow)**
- Prefer changing paths in **one place** (env vars / config YAML) and keep scripts generic.
- If you must edit Python eval scripts, do it once carefully (avoid repeated `sed` edits that can break indentation).

---

## 3) Step A — Train the Backdoored LoRA Adapter (Attack)

### 3.1 Pick a config
Example configs (names vary by repo):
- `configs/<task>/llama2_7b_chat/llama2_7b_<task>_<trigger>_lora.yaml`

Two common examples:
- `negsentiment + badnet`
- `refusal + badnet`

### 3.2 Verify key fields in the attack config YAML
You should check at least:
- `model_name_or_path: <BASE_MODEL_DIR>`
- `output_dir: backdoor_weight/<MODEL_NAME>/<TASK>/<TRIGGER>`

Example:
```yaml
model_name_or_path: /path/to/models/Llama-2-7b-chat-hf
output_dir: backdoor_weight/LLaMA2-7B-Chat/refusal/badnet
```

### 3.3 Submit training job
Use the scheduler to request a GPU and run:
```bash
torchrun --nproc_per_node=1 backdoor_train.py configs/<...>.yaml
```

### 3.4 Confirm training artifacts
After training finishes, confirm the LoRA adapter exists at the output directory and contains at least:
- `adapter_config.json`
- `adapter_model.safetensors` (or `pytorch_model.bin`)

Example:
```bash
ls attack/DPA/backdoor_weight/LLaMA2-7B-Chat/refusal/badnet/
```

---

## 4) Step B — Apply CROW Consistency Training (Defense)

### 4.1 Verify key fields in the consistency config YAML
Important fields:
- `model_name_or_path: <BASE_MODEL_DIR>`
- `adapter_name_or_path: <PATH_TO_BACKDOORED_LORA_OUTPUT>`
- `output_dir: backdoor_weight/<MODEL_NAME>/consistency/<TASK>_<TRIGGER>`

Example:
```yaml
model_name_or_path: /path/to/models/Llama-2-7b-chat-hf
adapter_name_or_path: /path/to/PROJECT_ROOT/attack/DPA/backdoor_weight/LLaMA2-7B-Chat/refusal/badnet
output_dir: backdoor_weight/LLaMA2-7B-Chat/consistency/refusal_badnet
```

### 4.2 Submit defense job
Run:
```bash
torchrun --nproc_per_node=1 consistency_train.py configs/consistency/llama2_7b_chat/<...>.yaml
```

### 4.3 Confirm defense artifacts
CROW outputs often include a *checkpoint folder*. Check the output directory for:
- `checkpoint-XX/adapter_config.json`
- `checkpoint-XX/adapter_model.safetensors`

Example:
```bash
find attack/DPA/backdoor_weight/LLaMA2-7B-Chat/consistency/refusal_badnet -maxdepth 2 -type f -name "adapter_config.json"
```

If you see `checkpoint-60/adapter_config.json`, then the LoRA path for evaluation should point to:
```
.../consistency/refusal_badnet/checkpoint-60
```

---

## 5) Step C — ASR Evaluation (Clean + Poison)

### 5.1 What ASR means
**ASR (Attack Success Rate)** is the percentage of *poisoned/triggered* test inputs for which the model exhibits the attacker’s target behavior.

Example interpretations:
- ASR ≈ 100%: backdoor is very effective
- ASR ≈ 0%: backdoor is mostly neutralized (good defense)

### 5.2 Fix evaluation script paths (critical)
Evaluation scripts typically contain fields like:
- `model_path` (base model directory)
- `tokenizer_path`
- `lora_model_path` (the LoRA adapter path)
- `test_clean_file`
- `test_trigger_file`
- `save_dir`

**Best practice:** use absolute paths and point `lora_model_path` to the **actual folder containing `adapter_config.json`** (often `checkpoint-XX` for defended models).

Example:
```python
task = {
  "model_path": "/path/to/models/Llama-2-7b-chat-hf",
  "tokenizer_path": "/path/to/models/Llama-2-7b-chat-hf",
  "use_lora": True,
  "lora_model_path": "/path/to/backdoor_weight/LLaMA2-7B-Chat/consistency/refusal_badnet/checkpoint-60",
  "test_clean_file": "/path/to/data/test_data/clean/refusal/test_data_no_trigger.json",
  "test_trigger_file": "/path/to/data/test_data/poison/refusal/badnet/backdoor200_refusal_badnet.json",
  "save_dir": "/path/to/eval_result/refusal/badnet/eval_LLaMA2-7B-Chat"
}
```

### 5.3 Run evaluation job
A typical job runs:
```bash
python3 backdoor_evaluate_refusal_badnet.py
```

The script should print something like:
- `Clean Evaluation`
- `Poison Evaluation`
- `ASR <value>`
and save a JSON results file under `save_dir`.

---

## 6) Common Pitfalls & Fixes

### 6.1 “LoRA model path does not exist”
Cause: relative path resolves incorrectly under the scheduler.
Fix:
- Use absolute `lora_model_path`
- Ensure it points to the folder that contains `adapter_config.json`
- For defended models, that may be `.../checkpoint-XX`

### 6.2 “Cannot find adapter_config.json”
Cause: pointing to parent directory instead of checkpoint directory.
Fix: update LoRA path to:
- `.../consistency/<task>_<trigger>/checkpoint-XX`

### 6.3 “FileNotFoundError: test_data_no_trigger.json”
Cause: test data paths are relative.
Fix: use absolute test data paths.

### 6.4 “IndentationError” after using sed
Cause: editing the file in a way that breaks Python indentation.
Fix:
- restore the file from version control (recommended), then reapply clean edits once
- avoid multiple overlapping `sed` replacements on already-edited files



## 7) Evaluate Without CROW (Baseline Comparison)

To compare defense effectiveness:
- Evaluate the **raw backdoored LoRA** (attack output) as `lora_model_path`
- Evaluate the **CROW-protected LoRA** (consistency checkpoint) as `lora_model_path`

Then compare:
- ASR(raw backdoored) vs ASR(CROW-protected)
- Clean behavior quality (qualitative outputs, or clean success metrics if provided)

---

## 8) Notes on Logging (Scheduler)

If you want job logs to always land in a dedicated folder:
- Use scheduler directives to set output/error directories PBS: `#PBS -o`, `#PBS -e`; 
- Also redirect Python output to your own `*.out` file if preferred.


# Appendix A — Example HPC Job Script (`.sh`) 

Save this file as `scripts/run_replication_slurm.sh`, then submit with:
```bash
qsub scripts/run_replication_slurm.sh
```



```bash
#!/usr/bin/env bash
#SBATCH --job-name=crow-repl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

############################################
# 0) Modules / conda 
############################################
# module purge
# module load cuda/12.1
# module load gcc/11.2

# If conda isn't auto-initialized in batch jobs, you may need:
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate crow

############################################
# 1) Paths (EDIT THESE)
############################################
export PROJECT_ROOT="/path/to/PROJECT_ROOT"
export BASE_MODEL_DIR="/path/to/models/Llama-2-7b-chat-hf"
export HF_HOME="/path/to/hf_cache"

# Attack / defense configs (EDIT THESE)
ATTACK_CFG="$PROJECT_ROOT/attack/DPA/configs/<task>/llama2_7b_chat/<attack_config>.yaml"
DEFENSE_CFG="$PROJECT_ROOT/attack/DPA/configs/consistency/llama2_7b_chat/<consistency_config>.yaml"

# Evaluation script (EDIT THIS)
EVAL_PY="$PROJECT_ROOT/attack/DPA/eval_scripts/backdoor_evaluate_refusal_badnet.py"

############################################
# 2) Housekeeping
############################################
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT/attack/DPA"


############################################
# 3) Step A — Attack training
############################################
echo "==== Step A: Backdoor LoRA training ===="
torchrun --nproc_per_node=1 backdoor_train.py "$ATTACK_CFG"

############################################
# 4) Step B — CROW consistency training
############################################
echo "==== Step B: CROW consistency training ===="
torchrun --nproc_per_node=1 consistency_train.py "$DEFENSE_CFG"

############################################
# 5) Step C — ASR evaluation
############################################
echo "==== Step C: ASR evaluation ===="
python3 "$EVAL_PY"

echo "==== DONE ===="
```