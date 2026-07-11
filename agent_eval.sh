#!/bin/bash
#SBATCH --job-name=webshop_rebuttal
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --gres=gpu:1
#SBATCH --output=logs/webshop_eval_%A_%a.out
#SBATCH --error=logs/webshop_eval_%A_%a.err

set -euo pipefail

# Keep attack checkpoints separate.  In particular, never run an indirect
# condition with the query-attack checkpoint.
QUERY_CKPT="/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118"
OBS_CKPT="/dataset/suaq0001/BackAgentDef/outputs/observation_attack/checkpoint-118"
CLEAN_CKPT="${CLEAN_CKPT:-$QUERY_CKPT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEBSHOP_DIR="${WEBSHOP_DIR:-$SCRIPT_DIR/agent-backdoor-attacks/AgentTuning/WebShop}"
if [[ "$WEBSHOP_DIR" != /* ]]; then
    WEBSHOP_DIR="$SCRIPT_DIR/$WEBSHOP_DIR"
fi
RESULTS_ROOT="${RESULTS_ROOT:-$WEBSHOP_DIR/results/rebuttal}"
GOAL_CACHE="${GOAL_CACHE:-$RESULTS_ROOT/goal_contract_cache.json}"

# Resolve path overrides before the eventual cd into WEBSHOP_DIR so directory
# creation, printed commands, and Python all refer to the same files.
if [[ "$RESULTS_ROOT" != /* ]]; then
    RESULTS_ROOT="$SCRIPT_DIR/$RESULTS_ROOT"
fi
if [[ "$GOAL_CACHE" != /* ]]; then
    GOAL_CACHE="$SCRIPT_DIR/$GOAL_CACHE"
fi

REBUTTAL_STAGE="${REBUTTAL_STAGE:-baselines}"
REBUTTAL_DRY_RUN="${REBUTTAL_DRY_RUN:-false}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-${REBUTTAL_TASK_ID:-0}}"
NUM_EVAL="${NUM_EVAL:-100}"
SEED="${SEED:-42}"
CLEAN_SPLIT="${CLEAN_SPLIT:-std}"
TARGET_BRAND="${TARGET_BRAND:-adidas}"

if [[ ! "$ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: array task ID must be a non-negative integer; got '$ARRAY_TASK_ID'." >&2
    exit 2
fi

DEFENSE=""
METHOD=""
METHOD_PATH=""
RUNTIME_MODE="none"
SETTING=""
ATTACK_SETTING=""
ORACLE_STRATEGY="none"
ORACLE_MODE="none"
MATRIX_SIZE=0

case "$REBUTTAL_STAGE" in
    baselines)
        MATRIX_SIZE=18
        if (( ARRAY_TASK_ID >= MATRIX_SIZE )); then
            echo "ERROR: baselines array index must be in 0-17; got $ARRAY_TASK_ID." >&2
            exit 2
        fi
        baseline_methods=(none legal_repair lexical_guard llm_judge goal_reminder gate)
        baseline_settings=(clean direct indirect)
        method_index=$((ARRAY_TASK_ID / 3))
        setting_index=$((ARRAY_TASK_ID % 3))
        DEFENSE="${baseline_methods[$method_index]}"
        ATTACK_SETTING="${baseline_settings[$setting_index]}"
        SETTING="$ATTACK_SETTING"
        if [[ "$DEFENSE" == "gate" ]]; then
            RUNTIME_MODE="full"
            METHOD="gate/full"
            METHOD_PATH="gate/full"
        else
            METHOD="$DEFENSE"
            METHOD_PATH="$DEFENSE"
        fi
        ;;
    mechanisms)
        MATRIX_SIZE=6
        if (( ARRAY_TASK_ID >= MATRIX_SIZE )); then
            echo "ERROR: mechanisms array index must be in 0-5; got $ARRAY_TASK_ID." >&2
            exit 2
        fi
        mechanism_modes=(mask_only enforce_only)
        mechanism_settings=(clean direct indirect)
        method_index=$((ARRAY_TASK_ID / 3))
        setting_index=$((ARRAY_TASK_ID % 3))
        DEFENSE="gate"
        RUNTIME_MODE="${mechanism_modes[$method_index]}"
        ATTACK_SETTING="${mechanism_settings[$setting_index]}"
        SETTING="$ATTACK_SETTING"
        METHOD="gate/$RUNTIME_MODE"
        METHOD_PATH="gate/$RUNTIME_MODE"
        ;;
    oracle)
        MATRIX_SIZE=12
        if (( ARRAY_TASK_ID >= MATRIX_SIZE )); then
            echo "ERROR: oracle array index must be in 0-11; got $ARRAY_TASK_ID." >&2
            exit 2
        fi
        oracle_methods=(none lexical_guard llm_judge gate)
        oracle_settings=(target_brand_direct target_brand_indirect near_miss_price_indirect)
        method_index=$((ARRAY_TASK_ID / 3))
        setting_index=$((ARRAY_TASK_ID % 3))
        DEFENSE="${oracle_methods[$method_index]}"
        SETTING="${oracle_settings[$setting_index]}"
        case "$setting_index" in
            0)
                ATTACK_SETTING="direct"
                ORACLE_STRATEGY="target_brand"
                ORACLE_MODE="direct_oracle"
                ;;
            1)
                ATTACK_SETTING="indirect"
                ORACLE_STRATEGY="target_brand"
                ORACLE_MODE="indirect_oracle"
                ;;
            2)
                ATTACK_SETTING="indirect"
                ORACLE_STRATEGY="near_miss_price"
                ORACLE_MODE="indirect_oracle"
                ;;
        esac
        if [[ "$DEFENSE" == "gate" ]]; then
            RUNTIME_MODE="full"
            METHOD="gate/full"
            METHOD_PATH="gate/full"
        else
            METHOD="$DEFENSE"
            METHOD_PATH="$DEFENSE"
        fi
        ;;
    *)
        echo "ERROR: REBUTTAL_STAGE must be baselines, mechanisms, or oracle; got '$REBUTTAL_STAGE'." >&2
        exit 2
        ;;
esac

TASK_IDS_PATH=""
case "$ATTACK_SETTING" in
    clean)
        EVAL_TYPE="clean"
        CHECKPOINT="$CLEAN_CKPT"
        TASK_IDS_PATH="${TEST_IDS_PATH:-${CLEAN_TEST_IDS_PATH:-}}"
        ;;
    direct)
        EVAL_TYPE="query_attack"
        CHECKPOINT="$QUERY_CKPT"
        TASK_IDS_PATH="${TEST_IDS_PATH:-${DIRECT_TEST_IDS_PATH:-$WEBSHOP_DIR/sneaker0_test_ids.json}}"
        ;;
    indirect)
        EVAL_TYPE="observation_attack"
        CHECKPOINT="$OBS_CKPT"
        TASK_IDS_PATH="${TEST_IDS_PATH:-${INDIRECT_TEST_IDS_PATH:-$WEBSHOP_DIR/sneakeri_test_ids.json}}"
        ;;
    *)
        echo "ERROR: internal unknown setting '$ATTACK_SETTING'." >&2
        exit 2
        ;;
esac

# Evaluation runs from WEBSHOP_DIR, so interpret relative task-ID overrides in
# the same place as test.py would interpret its PATH argument.
if [[ -n "$TASK_IDS_PATH" && "$TASK_IDS_PATH" != /* ]]; then
    TASK_IDS_PATH="$WEBSHOP_DIR/$TASK_IDS_PATH"
fi

OUTPUT_DIR="$RESULTS_ROOT/$REBUTTAL_STAGE/$METHOD_PATH"
OUTPUT_PATH="$OUTPUT_DIR/$SETTING.jsonl"
SUMMARY_PATH="$OUTPUT_DIR/$SETTING.summary.json"

PARSER_MODEL="not-applicable"
JUDGE_MODEL_RESOLVED="not-applicable"
if [[ "$DEFENSE" == "gate" ]]; then
    GATE_MODEL="${GATE_MODEL:?Set GATE_MODEL to the exact model used in the paper}"
    PARSER_MODEL="$GATE_MODEL"
fi
if [[ "$DEFENSE" == "llm_judge" ]]; then
    JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL to the exact trusted judge model}"
    JUDGE_MODEL_RESOLVED="$JUDGE_MODEL"
fi

CMD=(
    python test.py
    -c "$CHECKPOINT"
    -o "$OUTPUT_PATH"
    --type "$EVAL_TYPE"
    --gpu 0
    --defense "$DEFENSE"
    --num_eval "$NUM_EVAL"
    --seed "$SEED"
    --summary_path "$SUMMARY_PATH"
)

if [[ -n "$TASK_IDS_PATH" ]]; then
    CMD+=(--test_ids_path "$TASK_IDS_PATH")
else
    CMD+=(--clean_split "$CLEAN_SPLIT")
fi

if [[ "$DEFENSE" == "gate" ]]; then
    CMD+=(
        --gate_ablation full
        --gate_runtime_mode "$RUNTIME_MODE"
        --gate_openai_model "$GATE_MODEL"
        --require_goal_parser_success
        --goal_contract_cache "$GOAL_CACHE"
    )
fi

if [[ "$DEFENSE" == "llm_judge" ]]; then
    CMD+=(--judge_model "$JUDGE_MODEL")
fi

if [[ "$REBUTTAL_STAGE" == "oracle" ]]; then
    CMD+=(
        --stress_test action_oracle
        --oracle_mode "$ORACLE_MODE"
        --oracle_strategy "$ORACLE_STRATEGY"
        --target_brand "$TARGET_BRAND"
        --debug_log_full_text
    )
fi

print_resolution() {
    local task_ids_display="$TASK_IDS_PATH"
    if [[ -z "$task_ids_display" ]]; then
        task_ids_display="built-in clean split: $CLEAN_SPLIT"
    fi
    echo "================ RESOLVED MATRIX ROW ================"
    echo "Stage: $REBUTTAL_STAGE"
    echo "Array index: $ARRAY_TASK_ID / $((MATRIX_SIZE - 1))"
    echo "Method: $METHOD"
    echo "Defense: $DEFENSE"
    echo "Runtime mode: $RUNTIME_MODE"
    echo "Setting: $SETTING"
    echo "Evaluation type: $EVAL_TYPE"
    echo "Oracle strategy: $ORACLE_STRATEGY"
    echo "Checkpoint: $CHECKPOINT"
    echo "Task IDs: $task_ids_display"
    echo "Output: $OUTPUT_PATH"
    echo "Summary: $SUMMARY_PATH"
    echo "Parser model: $PARSER_MODEL"
    echo "Judge model: $JUDGE_MODEL_RESOLVED"
    echo "Goal cache: $GOAL_CACHE"
    printf 'Command: cd %q &&' "$WEBSHOP_DIR"
    printf ' %q' "${CMD[@]}"
    printf '\n'
    echo "====================================================="
}

print_resolution

case "${REBUTTAL_DRY_RUN,,}" in
    true|1|yes)
        echo "REBUTTAL_DRY_RUN=true; evaluation was not invoked."
        exit 0
        ;;
    false|0|no)
        ;;
    *)
        echo "ERROR: REBUTTAL_DRY_RUN must be true or false; got '$REBUTTAL_DRY_RUN'." >&2
        exit 2
        ;;
esac

# Fail before environment setup or model loading when a selected method needs
# the trusted external API.  Non-GATE/non-judge rows do not require an API key.
if [[ "$DEFENSE" == "gate" || "$DEFENSE" == "llm_judge" ]]; then
    export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY must be exported}"
fi

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR" "$(dirname -- "$GOAL_CACHE")"

echo "================ BASIC JOB INFO ================"
echo "DATE: $(date)"
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES assigned by Slurm: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-unset}"
echo "================================================"

export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
mkdir -p "$TMPDIR"

CONDA_SH="${CONDA_SH:-/export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-webshop_torchfix}"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda initialization script not found: $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ ! -d "$WEBSHOP_DIR" ]]; then
    echo "ERROR: WebShop directory not found: $WEBSHOP_DIR" >&2
    exit 1
fi
if [[ ! -f "$WEBSHOP_DIR/test.py" ]]; then
    echo "ERROR: evaluator not found: $WEBSHOP_DIR/test.py" >&2
    exit 1
fi
if [[ -n "$TASK_IDS_PATH" && ! -f "$TASK_IDS_PATH" ]]; then
    echo "ERROR: task-ID file not found: $TASK_IDS_PATH" >&2
    exit 1
fi

cd "$WEBSHOP_DIR"
"${CMD[@]}"
