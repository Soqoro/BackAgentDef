#!/bin/bash
#SBATCH --job-name=webshop-choice-integrity
#SBATCH --partition=NA100q
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-17%2
#SBATCH --output=choice_integrity-%A_%a.out
#SBATCH --error=choice_integrity-%A_%a.err
#SBATCH --export=ALL

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

die() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARNING: $*" >&2
}

as_bool() {
  case "${1,,}" in
    1|true|yes|on) echo 1 ;;
    0|false|no|off|"") echo 0 ;;
    *) die "Expected a boolean value, got '$1'." ;;
  esac
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

resolve_repo_root() {
  local candidate
  local candidates=()

  if [[ -n "${REPO_ROOT:-}" ]]; then
    candidates+=("$REPO_ROOT")
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    candidates+=("$SLURM_SUBMIT_DIR")
  fi
  candidates+=("$PWD")
  candidates+=("/export/home2/suaq0001/BackAgentDef")

  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate/agent-backdoor-attacks/AgentTuning/WebShop" ]]; then
      (
        cd "$candidate"
        pwd -P
      )
      return 0
    fi
  done

  echo "Could not locate the BackAgentDef repository." >&2
  echo "Checked, in order:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  echo "Set REPO_ROOT to the repository root when submitting." >&2
  return 1
}

DRY_RUN="$(as_bool "${DRY_RUN:-0}")"
WEBSHOP_USE_CATALOG_RATINGS="$(
  as_bool "${WEBSHOP_USE_CATALOG_RATINGS:-1}"
)"
export WEBSHOP_USE_CATALOG_RATINGS
SMOKE_NUM_TASKS="${SMOKE_NUM_TASKS:-0}"
SEED="${SEED:-42}"
MATRIX_INDEX="${SLURM_ARRAY_TASK_ID:-${MATRIX_INDEX:-0}}"

[[ "$SMOKE_NUM_TASKS" =~ ^[0-9]+$ ]] \
  || die "SMOKE_NUM_TASKS must be a non-negative integer."
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer."
[[ "$MATRIX_INDEX" =~ ^[0-9]+$ ]] || die "Matrix index must be an integer in [0, 17]."
(( MATRIX_INDEX >= 0 && MATRIX_INDEX < 18 )) \
  || die "Matrix index $MATRIX_INDEX is outside the 18-cell range [0, 17]."

RESOLVED_REPO_ROOT="$(resolve_repo_root)" || exit 1
WEBSHOP_ROOT="$RESOLVED_REPO_ROOT/agent-backdoor-attacks/AgentTuning/WebShop"
RUNNER="$WEBSHOP_ROOT/choice_integrity_eval.py"

CONFIG_PATH="${CONFIG_PATH:-$WEBSHOP_ROOT/choice_integrity/config.default.json}"
MANIFEST_PATH="${MANIFEST_PATH:-$WEBSHOP_ROOT/benchmarks/choice_integrity_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dataset/suaq0001/BackAgentDef/outputs/choice_integrity}"

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"
CHECKPOINT_PATH="${MODEL_CHECKPOINT:-<MODEL_CHECKPOINT-required>}"

METHODS=(
  "undefended"
  "gate"
  "state_aware_verifier"
  "gate_ci"
  "gate_ci_no_ledger"
  "gate_ci_no_dominance"
)
CONDITIONS=(
  "clean"
  "direct"
  "indirect"
)

METHOD_INDEX=$((MATRIX_INDEX / ${#CONDITIONS[@]}))
CONDITION_INDEX=$((MATRIX_INDEX % ${#CONDITIONS[@]}))
METHOD="${METHODS[$METHOD_INDEX]}"
CONDITION="${CONDITIONS[$CONDITION_INDEX]}"

if [[ -n "${RUN_ID:-}" ]]; then
  RESOLVED_RUN_ID="$RUN_ID"
elif [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  RESOLVED_RUN_ID="slurm_${SLURM_ARRAY_JOB_ID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RESOLVED_RUN_ID="slurm_${SLURM_JOB_ID}"
else
  RESOLVED_RUN_ID="manual_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
export RUN_ID="$RESOLVED_RUN_ID"

RUN_DIR="$OUTPUT_ROOT/runs/$RESOLVED_RUN_ID"
CELL_DIR="$RUN_DIR/cells/$METHOD/$CONDITION/seed_$SEED"
AGGREGATE_DIR="$RUN_DIR/aggregate"
AGGREGATE_LOCK="$RUN_DIR/.aggregate.lock"

REQUIRE_OPENAI="${REQUIRE_OPENAI:-auto}"
if [[ "$REQUIRE_OPENAI" == "auto" ]]; then
  case "$METHOD" in
    undefended) REQUIRE_OPENAI=0 ;;
    gate|state_aware_verifier|gate_ci|gate_ci_no_ledger|gate_ci_no_dominance)
      REQUIRE_OPENAI=1
      ;;
    *) die "Internal error: unsupported method '$METHOD'." ;;
  esac
else
  REQUIRE_OPENAI="$(as_bool "$REQUIRE_OPENAI")"
fi

BUILD_COMMAND=(
  python -u choice_integrity_eval.py build
  --config "$CONFIG_PATH"
  --manifest "$MANIFEST_PATH"
)

RUN_COMMAND=(
  python -u choice_integrity_eval.py run
  --manifest "$MANIFEST_PATH"
  --config "$CONFIG_PATH"
  --method "$METHOD"
  --condition "$CONDITION"
  --checkpoint "$CHECKPOINT_PATH"
  --output-dir "$CELL_DIR"
  --seed "$SEED"
)

if (( SMOKE_NUM_TASKS > 0 )); then
  RUN_COMMAND+=(--num-tasks "$SMOKE_NUM_TASKS")
fi

AGGREGATE_COMMAND=(
  python -u choice_integrity_eval.py aggregate
  --manifest "$MANIFEST_PATH"
  --config "$CONFIG_PATH"
  --run-dir "$RUN_DIR"
  --output-dir "$AGGREGATE_DIR"
)

echo "================ CHOICE-INTEGRITY CELL ================"
echo "Date: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Repository: $RESOLVED_REPO_ROOT"
echo "WebShop: $WEBSHOP_ROOT"
echo "Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Slurm array job: ${SLURM_ARRAY_JOB_ID:-manual}"
echo "Array index: $MATRIX_INDEX"
echo "Method: $METHOD"
echo "Condition: $CONDITION"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Manifest: $MANIFEST_PATH"
echo "Config: $CONFIG_PATH"
echo "Run directory: $RUN_DIR"
echo "Cell directory: $CELL_DIR"
echo "Smoke task limit: $SMOKE_NUM_TASKS"
echo "Catalog ratings enabled: $WEBSHOP_USE_CATALOG_RATINGS"
echo "OpenAI required: $REQUIRE_OPENAI"
echo "Dry run: $DRY_RUN"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<set by Slurm>}"
echo "======================================================="

if (( DRY_RUN == 1 )); then
  echo "DRY_RUN=1; filesystem, environment, import, and GPU preflights are skipped."
  if [[ ! -f "$MANIFEST_PATH" ]]; then
    warn "Frozen manifest is absent. Build it once with:"
    echo "  cd '$WEBSHOP_ROOT'"
    print_command "${BUILD_COMMAND[@]}"
  fi
  echo "Run command:"
  print_command "${RUN_COMMAND[@]}"
  echo "Aggregate command (normally executed under flock):"
  print_command "${AGGREGATE_COMMAND[@]}"
  exit 0
fi

[[ -f "$RUNNER" ]] || die "Runner not found: $RUNNER"
[[ -f "$CONFIG_PATH" ]] || die "Choice-integrity config not found: $CONFIG_PATH"
[[ -n "$MODEL_CHECKPOINT" ]] || die \
  "MODEL_CHECKPOINT must name the single combined compromised checkpoint used by every paired condition. Export it before sbatch."

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "The frozen choice-integrity manifest does not exist:" >&2
  echo "  $MANIFEST_PATH" >&2
  echo "Build and audit it once before submitting evaluation jobs:" >&2
  echo "  cd '$WEBSHOP_ROOT'" >&2
  print_command "${BUILD_COMMAND[@]}" >&2
  echo "The evaluation launcher will never build or modify the manifest implicitly." >&2
  exit 2
fi

[[ -d "$CHECKPOINT_PATH" ]] || die "Checkpoint directory not found: $CHECKPOINT_PATH"
[[ -f "$CHECKPOINT_PATH/config.json" ]] \
  || die "Checkpoint is missing config.json: $CHECKPOINT_PATH"
[[ -f "$CHECKPOINT_PATH/choice_integrity_provenance.json" ]] \
  || die "Checkpoint is missing choice_integrity_provenance.json; use the jointly trained direct+indirect checkpoint documented in choice_integrity/README.md."

DATA_DIR="$WEBSHOP_ROOT/data"
INDEX_DIR="$WEBSHOP_ROOT/search_engine/indexes"
for required_data_file in \
  "$DATA_DIR/items_shuffle.json" \
  "$DATA_DIR/items_ins_v2.json" \
  "$DATA_DIR/items_human_ins.json"
do
  [[ -f "$required_data_file" ]] || die "Required WebShop data file not found: $required_data_file"
done

[[ -d "$INDEX_DIR" ]] || die "WebShop Lucene index directory not found: $INDEX_DIR"
compgen -G "$INDEX_DIR/segments_*" >/dev/null \
  || die "No Lucene segments_* file found under $INDEX_DIR"

command -v flock >/dev/null || die "flock is required for safe concurrent aggregation."
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable on this compute node."

CONDA_SH="${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-webshop_torchfix}"
[[ -f "$CONDA_SH" ]] || die "Conda activation script not found: $CONDA_SH"

export CONDA_NO_PLUGINS=true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"
unset PYTHONPATH || true

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$WEBSHOP_ROOT"

command -v java >/dev/null || die "Java is required by Pyserini/Lucene."

if (( REQUIRE_OPENAI == 1 )) && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  die "OPENAI_API_KEY is required for method '$METHOD'. Supply it through your cluster's secret/environment mechanism; never put it in this script."
fi

export CI_REQUIRE_OPENAI="$REQUIRE_OPENAI"

cd "$WEBSHOP_ROOT"

python - "$REQUIRE_OPENAI" <<'PY'
import importlib
import sys

required = ["torch", "transformers", "fastchat", "gym", "pyserini", "choice_integrity"]
if sys.argv[1] == "1":
    required.append("openai")

failures = []
for name in required:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        location = getattr(module, "__file__", "built-in")
        print(f"{name}: version={version} path={location}")
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")

if failures:
    raise SystemExit("Required import failures:\n  " + "\n  ".join(failures))
PY

python - <<'PY'
import os
import torch

print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("visible GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("Slurm allocated no usable CUDA GPU.")

print("logical cuda:0:", torch.cuda.get_device_name(0))
PY

mkdir -p "$CELL_DIR" "$AGGREGATE_DIR"

echo "Running:"
print_command "${RUN_COMMAND[@]}"
"${RUN_COMMAND[@]}" 2>&1 | tee "$CELL_DIR/run.log"

echo "Aggregating available completed cells under $AGGREGATE_LOCK"
(
  flock -x 9
  echo "Aggregate command:"
  print_command "${AGGREGATE_COMMAND[@]}"
  "${AGGREGATE_COMMAND[@]}" 2>&1 | tee -a "$RUN_DIR/aggregate.log"
) 9>"$AGGREGATE_LOCK"

echo "Completed matrix cell $MATRIX_INDEX: method=$METHOD condition=$CONDITION"
