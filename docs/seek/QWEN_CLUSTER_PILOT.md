# Local Qwen defender: cluster pilot

This setup uses the existing Qwen3.5-27B snapshot at revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`. The user reported that all locked files
exist. This is not yet a hash verification or a successful model load. The original
lock status is `staged_not_gpu_validated`; the implementation never rewrites it.
No GPU work, model download, paid API, training or Slurm submission was performed
while implementing this adapter.

## Resource and software evidence

The supplied `docs/AAI-resourcelist.pdf`, page 2, lists PH100q/node06 as H100 80GB
and NH100q/node07 as H100 80GB. PA10080q/node04 and NA100q/node01 are A100 80GB;
PA100q and NA10040q are A100 40GB. RTXA6Kq has 48GB cards. HPCAIq prioritizes
CPU-intensive workloads. The supplied user guide, pages 4–5, requires explicit
GPU allocation and recommends four CPU cores per GPU.

The user's queue snapshot showed four unallocated GPUs on PH100q, with NA100q
draining and NH100q/PA10080q fully allocated. Availability and partition access
must be checked at submission; those observations are not reservations.

Use one PH100q GPU for the defender-only smoke, then two for discovery: visible
cuda:0 for the legacy victim, visible cuda:1 for Qwen. Never hard-code physical
GPU indices or overwrite Slurm's CUDA_VISIBLE_DEVICES. The roughly 55.6GB
checkpoint does not fit unsharded on a 40GB/48GB GPU; 80GB leaves some runtime
headroom, but only a real load/generation can establish memory sufficiency.

The reported webshop_torchfix environment has torch 2.10.0+cu128,
Transformers 4.57.6, Accelerate 1.13.0, and no vLLM. Native Qwen3.5 support is
available in [Transformers 5.6.2](https://huggingface.co/docs/transformers/v5.6.2/model_doc/qwen3_5).
Use a separate environment; do not upgrade the victim's Transformers.
The [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-27B) documents disabling
thinking through the chat template. This pilot uses that setting, greedy decoding,
BF16, SDPA, at most 8192 input tokens and 1024 generated tokens. These are pilot
settings, not a claim that they maximize Qwen performance.

## 1. CPU setup and compatibility check (cluster login/Jupyter)

Run from your cluster checkout (`~/BackAgentDef` in the supplied terminal), while
webshop_torchfix is active. The venv inherits its installed Torch to avoid duplicating
large CUDA packages. Installing the pinned Transformers package below affects only
the new venv and may fetch Python packages; it does not download model weights.
If cluster package access is unavailable, install the same packages from your approved
wheel mirror. Do not modify the existing environment as a workaround.

```bash
cd ~/BackAgentDef
export SEEK_REPO_ROOT="$PWD"
export SEEK_QWEN_ENV=/dataset/suaq0001/seek-envs/qwen35-tf562
export SEEK_QWEN_PYTHON="$SEEK_QWEN_ENV/bin/python"
export SEEK_QWEN_LOCK=/dataset/suaq0001/beyond-consensus/models/Qwen--Qwen3.5-27B/model-lock.json
export SEEK_QWEN_AGENTS="$PWD/configs/seek/local/qwen_agents.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SEEK_AGENT_MODEL

# Create once; if this path already exists, inspect it instead of recreating it.
python -m venv --system-site-packages "$SEEK_QWEN_ENV"
"$SEEK_QWEN_PYTHON" -m pip install 'transformers==5.6.2'

# Imports the model class and reads local config/tokenizer; no weights or GPU work.
"$SEEK_QWEN_PYTHON" agent-backdoor-attacks/AgentTuning/WebShop/seek/qwen_worker.py \
  --check-lock "$SEEK_QWEN_LOCK" --check-imports

python docs/seek/prepare_qwen.py \
  --lock "$SEEK_QWEN_LOCK" --python "$SEEK_QWEN_PYTHON" \
  --output "$SEEK_QWEN_AGENTS"
"$SEEK_QWEN_PYTHON" -m pip freeze > "$SEEK_QWEN_ENV/seek-packages.txt"
```

The checker validates metadata hashes, architecture, tokenizer binding, all shard
paths and index coverage. Add `--verify-weights` for a CPU/I/O-only full shard hash
pass (~55.6GB read). Actual GPU workers always verify all hashes before loading.
An import failure is a dependency blocker; do not proceed to GPU submission.
The two environments must remain fixed throughout a run; create a new run ID after
any dependency/model/generation change. No automatic fallback or quantization occurs.

## 2. One-GPU smoke (user submits after the CPU check passes)

```bash
mkdir -p logs/seek
export SEEK_QWEN_MODE=smoke
SEEK_DRY_RUN=1 bash seek_qwen.sh
sbatch --partition=PH100q --gres=gpu:1 --cpus-per-task=4 \
  --mem=96G --time=00:30:00 seek_qwen.sh
```

Read `logs/seek/qwen-JOBID.out` and `.err`. Success is one JSON result with
`status: passed`, `test: local_qwen_json_smoke`, and `simulated: false`.
This proves a local model load and one small JSON generation only. It does not
verify all Goal/State/Action roles, causality, replay, or the victim checkpoint.
The smoke returns nonzero on truncation, invalid JSON, wrong content, import/load
failure, missing allocation, or insufficient visible GPU memory.

## 3. Freeze the Qwen settings before collection

Use the existing inventory → task manifest → collection → replay sequence in
`SEEK_RUNBOOK.md`. Substitute this setup command for its API-model setup command:

```bash
python docs/seek/prepare_cluster.py \
  --product-file "$PWD/agent-backdoor-attacks/AgentTuning/WebShop/data/items_shuffle.json" \
  --agent-config "$SEEK_QWEN_AGENTS" --output-dir configs/seek/local
python seek_eval.py preflight --config configs/seek/local/inventory.json \
  --phase inventory --metadata-only
export SEEK_RUN_ROOT="$PWD/results/seek"
export SBATCH_PARTITION=PH100q
export SBATCH_CPUS_PER_TASK=4
```

Checkpoint paths default to those observed in agent_eval.sh; supply the explicit
`--query-checkpoint` and `--observation-checkpoint` paths if different. Missing
WebShop assets, victim checkpoints and training provenance remain separate blockers.
The prepare helper emits immutable files. If an earlier API-configured pilot already
exists, use a new output directory and new run IDs; do not overwrite frozen configs.
The default pilot path assumes a fresh setup.

Inventory/collection/replay need no defender API credential or Qwen model load.
The SBATCH environment above changes the existing submission helper's partition
without editing agent_eval.sh or overriding its protected behavior. Follow the
runbook's inventory completion and manifest construction steps before collection.

## 4. Bounded local discovery (after successful replay review)

```bash
export SEEK_CONFIG="$PWD/configs/seek/local/cluster_pilot.json"
export SEEK_PHASE=discover
export SEEK_QWEN_MODE=discover
export SEEK_RUN_ROOT="$PWD/results/seek"
# Supply CONDA_SH if it differs from the default documented in SEEK_RUNBOOK.md.
export CONDA_ENV=webshop_torchfix
SEEK_DRY_RUN=1 bash seek_qwen.sh
sbatch --partition=PH100q --nodes=1 --ntasks=1 \
  --gres=gpu:2 --cpus-per-task=8 --mem=160G --time=02:00:00 \
  --array=0-1%1 --output='logs/seek/qwen-discover-%A_%a.out' \
  --error='logs/seek/qwen-discover-%A_%a.err' seek_qwen.sh
```

The parent stays in webshop_torchfix. Only the defender process uses the new Python.
All three roles share one resident Qwen model but send independent messages without
retained chat/KV state. The child uses no network serving endpoint and no API key.
The parent starts it lazily inside an accounted defender attempt, bounds startup
and per-call time, and kills/reaps it on timeout or completion. If it restarts on a
bounded retry, it verifies the lock again. Runtime package versions and model/lock
identity accompany the journaled response; failed attempts remain in the journal.

JSON is prompted and then validated against the existing role schema and evidence
checks; generation is not grammar-constrained. Invalid replies, refusals expressed
outside the schema, token-limit endings and timeouts remain backend failures after
bounded retries. Long prompts are rejected rather than silently truncated.
No discovery or confirmation thresholds/budgets are relaxed for this backend.

The conservative real WebShop source adapter still marks page text as protected.
A `no_valid_intervention` result is possible and is not proof that the victim is
clean. Independently audited narrative sources, replay validity and checkpoint/
training-overlap provenance are still prerequisites for causal confirmation.

## Local verification

The new tests use synthetic files and a tiny fake subprocess, not Qwen inference.
They cover hash/index pinning, corruption/missing files, configuration, allocation
requirements, persistent protocol, timeout cleanup, metadata-only imports and shell
dry runs. Real Qwen compatibility, memory, speed and role-quality verification are
pending the cluster commands above. Do not describe CPU test success as a real-model
result.

Verification of this change: 71 Seek CPU tests passed (including 12 new local-Qwen
synthetic tests), plus 52 existing Stage I Python tests and all 36 agent_eval.sh
Slurm dry-run rows. Python compilation, bash syntax and git whitespace checks
passed. Protected agent_eval.sh, WebShop/test.py and defenses files have no diff.
No real model was loaded and no job was submitted during these checks.

## Inventory import failure from inherited Jupyter paths

Reported cluster job 1083058_0 failed importing Flask: Python 3.10 Flask from
webshop_torchfix imported Werkzeug from `/cm/shared/apps/jupyter/16.0.5/lib/python3.12/site-packages`.
Metadata preflight was ready; inventory did not complete. This is an environment
import failure, not a model/provenance finding. Do not upgrade Flask, Gym or the
victim environment in response to the surrounding warnings before checking paths.

Seek entry scripts now unset PYTHONPATH/PYTHONHOME and set PYTHONNOUSERSITE=1 before
Python or conda runs. The worker repeats this after activation, and the isolated
Qwen subprocess also strips these inherited variables. Slurm GPU assignments stay
unchanged. CPU tests inject broken paths and an activation hook to verify this.

For an already-started run on the previous source revision, first try this
submission-environment repair without updating source files:

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 CONDA_NO_PLUGINS=true
python - <<'PY'
import sys
import flask
import werkzeug
print('Python:', sys.executable)
print('Flask:', flask.__file__)
print('Werkzeug:', werkzeug.__file__)
PY
```

If imports succeed from the intended environment, resubmit the same inventory
config with seek_submit.sh (which enables resume). If imports still fail, inspect
the new traceback before changing packages. Keep the failed logs and run directory.
The code fix changes the source fingerprint: after syncing it, use new run IDs for
inventory and the pilot instead of overwriting/resuming a run pinned to older source.
