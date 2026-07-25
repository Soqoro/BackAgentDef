#!/bin/bash
#SBATCH --job-name=choice-integrity-build
#SBATCH --partition=NA100q
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=choice-integrity-build-%j.out
#SBATCH --error=choice-integrity-build-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

die() {
  echo "ERROR: $*" >&2
  exit 1
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
    if [[ -f "$candidate/choice_integrity_eval.sh" ]] \
      && [[ -d "$candidate/agent-backdoor-attacks/AgentTuning/WebShop" ]]
    then
      (
        cd "$candidate"
        pwd -P
      )
      return 0
    fi
  done

  echo "Could not locate the BackAgentDef repository." >&2
  echo "Set REPO_ROOT explicitly when submitting this job." >&2
  return 1
}

RESOLVED_REPO_ROOT="$(resolve_repo_root)" || exit 1
WEBSHOP_ROOT="$RESOLVED_REPO_ROOT/agent-backdoor-attacks/AgentTuning/WebShop"
RUNNER="$WEBSHOP_ROOT/choice_integrity_eval.py"
CONFIG_PATH="${CONFIG_PATH:-$WEBSHOP_ROOT/choice_integrity/config.default.json}"
MANIFEST_PATH="${MANIFEST_PATH:-$WEBSHOP_ROOT/benchmarks/choice_integrity_v1.json}"
BUILD_REPORT_PATH="$MANIFEST_PATH.build_report.json"

CONDA_SH="${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-webshop_torchfix}"

[[ -f "$RUNNER" ]] || die "Runner not found: $RUNNER"
[[ -f "$CONFIG_PATH" ]] || die "Config not found: $CONFIG_PATH"
[[ -f "$CONDA_SH" ]] || die "Conda initialization script not found: $CONDA_SH"
[[ ! -e "$MANIFEST_PATH" ]] || die \
  "Manifest already exists; refusing to overwrite it: $MANIFEST_PATH"

DATA_DIR="$WEBSHOP_ROOT/data"
INDEX_DIR="$WEBSHOP_ROOT/search_engine/indexes"
for required_data_file in \
  "$DATA_DIR/items_shuffle.json" \
  "$DATA_DIR/items_ins_v2.json" \
  "$DATA_DIR/items_human_ins.json"
do
  [[ -f "$required_data_file" ]] \
    || die "Required WebShop data file not found: $required_data_file"
done
[[ -d "$INDEX_DIR" ]] || die "WebShop Lucene index directory not found: $INDEX_DIR"
compgen -G "$INDEX_DIR/segments_*" >/dev/null \
  || die "No Lucene segments_* file found under $INDEX_DIR"
[[ -x /usr/bin/time ]] || die "/usr/bin/time is required."

export CONDA_NO_PLUGINS=true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WEBSHOP_USE_CATALOG_RATINGS="${WEBSHOP_USE_CATALOG_RATINGS:-1}"
mkdir -p "$TMPDIR" "$(dirname "$MANIFEST_PATH")"
unset PYTHONPATH || true

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$WEBSHOP_ROOT"
command -v java >/dev/null || die "Java is required by Pyserini/Lucene."

cd "$WEBSHOP_ROOT"

echo "================ CHOICE-INTEGRITY BUILD ================"
echo "Date: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Repository: $RESOLVED_REPO_ROOT"
echo "Conda environment: $CONDA_ENV"
echo "Config: $CONFIG_PATH"
echo "Manifest: $MANIFEST_PATH"
echo "Catalog ratings enabled: $WEBSHOP_USE_CATALOG_RATINGS"
echo "This job requests no GPU."
echo "========================================================"

python - <<'PY'
import sys
from importlib.metadata import version

import flask
import werkzeug

print("python:", sys.executable)
print("flask:", version("Flask"), flask.__file__)
print("werkzeug:", version("Werkzeug"), werkzeug.__file__)
PY

/usr/bin/time -v python -u choice_integrity_eval.py build \
  --config "$CONFIG_PATH" \
  --manifest "$MANIFEST_PATH"

[[ -s "$MANIFEST_PATH" ]] || die "Build did not create a non-empty manifest."
[[ -s "$BUILD_REPORT_PATH" ]] || die "Build did not create its report."

echo "Build completed successfully."
sha256sum "$MANIFEST_PATH"
ls -lh "$MANIFEST_PATH" "$BUILD_REPORT_PATH"
