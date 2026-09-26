#!/usr/bin/env bash
set -euo pipefail

# Jupyter may export another Python installation into Slurm jobs.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEEK_CONFIG="" SEEK_PHASE="" SEEK_RUN_ROOT="$SOURCE_DIR/results/seek"
concurrency=1 dependency="" selected_row="" dry_run=0
while (($#)); do
    case "$1" in
        --config) SEEK_CONFIG="${2:?}"; shift 2 ;;
        --phase) SEEK_PHASE="${2:?}"; shift 2 ;;
        --run-root) SEEK_RUN_ROOT="${2:?}"; shift 2 ;;
        --max-concurrency) concurrency="${2:?}"; shift 2 ;;
        --dependency) dependency="${2:?}"; shift 2 ;;
        --row) selected_row="${2:?}"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$SEEK_CONFIG" || -z "$SEEK_PHASE" || ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: seek_submit.sh --config FILE --phase PHASE [--run-root DIR] [--row N] [--max-concurrency N] [--dependency afterok:JOB] [--dry-run]' >&2
    exit 2
fi
SEEK_CONFIG="$(realpath -- "$SEEK_CONFIG")"
SEEK_RUN_ROOT="$(realpath -m -- "$SEEK_RUN_ROOT")"
SEEK_PYTHON="${SEEK_PYTHON:-python}"
schedule=("$SEEK_PYTHON" "$SOURCE_DIR/seek_eval.py" schedule --config "$SEEK_CONFIG" --phase "$SEEK_PHASE" --run-root "$SEEK_RUN_ROOT")
if [[ -n "$selected_row" ]]; then schedule+=(--row "$selected_row"); fi
"${schedule[@]}"
count="$("$SEEK_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["rows"]))' "$SEEK_CONFIG")"
array="0-$((count - 1))%$concurrency"
if [[ -n "$selected_row" ]]; then array="$selected_row%$concurrency"; fi
log_dir="$SOURCE_DIR/logs/seek"
# Prepared before submission, including dry run. No conda/API/model initialization.
mkdir -p "$log_dir"
cmd=(sbatch --parsable --array="$array" --output="$log_dir/%A_%a.out" --error="$log_dir/%A_%a.err")
if [[ -n "$dependency" ]]; then
    if [[ ! "$dependency" =~ ^afterok:[0-9]+(:[0-9]+)*$ ]]; then echo 'Invalid dependency' >&2; exit 2; fi
    cmd+=(--dependency="$dependency")
fi
cmd+=(--export=ALL "$SOURCE_DIR/seek_eval.sh")
printf '%q ' "${cmd[@]}"
printf '\n'
if ((dry_run)); then exit 0; fi
export SEEK_CONFIG SEEK_PHASE SEEK_RUN_ROOT SEEK_RESUME=1
export SEEK_REPO_ROOT="$SOURCE_DIR"
"${cmd[@]}"
