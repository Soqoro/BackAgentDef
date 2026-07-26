#!/bin/bash

set -euo pipefail

TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TEST_DIR/../../../.." && pwd)"
HARNESS="$REPO_ROOT/agent_eval.sh"

QUERY_CKPT="/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118"
OBS_CKPT="/dataset/suaq0001/BackAgentDef/outputs/observation_attack/checkpoint-118"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

contains() {
    local text="$1"
    local expected="$2"
    [[ "$text" == *"$expected"* ]] || fail "missing expected text: $expected"
}

not_contains() {
    local text="$1"
    local unexpected="$2"
    [[ "$text" != *"$unexpected"* ]] || fail "found unexpected text: $unexpected"
}

# An exported shell function shadows the executable.  If a dry run invokes
# Python for any reason, the marker below makes the test fail.
python() {
    echo "DRY_RUN_INVOKED_PYTHON" >&2
    return 97
}
export -f python

bash -n "$HARNESS"
grep -q '^#SBATCH --gres=gpu:1$' "$HARNESS" || fail "one GPU is not requested"
cuda_assignments="$(grep -En '^[[:space:]]*(export[[:space:]]+)?CUDA_VISIBLE_DEVICES=' "$HARNESS" || true)"
[[ "$cuda_assignments" == *'export CUDA_VISIBLE_DEVICES="$SELECTED_CUDA_DEVICE"'* ]] || \
    fail "agent_eval.sh lacks the validated physical-GPU assignment"
[[ "$(wc -l <<<"$cuda_assignments")" -eq 1 ]] || \
    fail "agent_eval.sh has an unexpected CUDA_VISIBLE_DEVICES assignment"

declare -A seen_outputs=()
declare -A seen_summaries=()
checked=0
gate_rows=0
judge_rows=0

for stage_spec in baselines:18 mechanisms:6 oracle:12; do
    stage="${stage_spec%%:*}"
    count="${stage_spec##*:}"

    for ((index = 0; index < count; index++)); do
        output="$({
            env -u OPENAI_API_KEY \
                REBUTTAL_STAGE="$stage" \
                SLURM_ARRAY_TASK_ID="$index" \
                REBUTTAL_DRY_RUN=true \
                GATE_MODEL=test-gate-model \
                JUDGE_MODEL=test-judge-model \
                bash "$HARNESS"
        } 2>&1)" || fail "$stage[$index] dry run failed"

        not_contains "$output" "DRY_RUN_INVOKED_PYTHON"
        contains "$output" "Stage: $stage"
        contains "$output" "Array index: $index /"
        contains "$output" "REBUTTAL_DRY_RUN=true; evaluation was not invoked."

        method="$(sed -n 's/^Method: //p' <<<"$output")"
        defense="$(sed -n 's/^Defense: //p' <<<"$output")"
        runtime_mode="$(sed -n 's/^Runtime mode: //p' <<<"$output")"
        setting="$(sed -n 's/^Setting: //p' <<<"$output")"
        eval_type="$(sed -n 's/^Evaluation type: //p' <<<"$output")"
        checkpoint="$(sed -n 's/^Checkpoint: //p' <<<"$output")"
        task_ids="$(sed -n 's/^Task IDs: //p' <<<"$output")"
        result_path="$(sed -n 's/^Output: //p' <<<"$output")"
        summary_path="$(sed -n 's/^Summary: //p' <<<"$output")"
        command_line="$(sed -n 's/^Command: //p' <<<"$output")"

        [[ -n "$command_line" ]] || fail "$stage[$index] did not print a command"
        [[ -z "${seen_outputs[$result_path]+present}" ]] || fail "duplicate output: $result_path"
        [[ -z "${seen_summaries[$summary_path]+present}" ]] || fail "duplicate summary: $summary_path"
        seen_outputs["$result_path"]=1
        seen_summaries["$summary_path"]=1

        case "$stage" in
            baselines)
                methods=(none legal_repair lexical_guard llm_judge goal_reminder gate/full)
                settings=(clean direct indirect)
                expected_method="${methods[$((index / 3))]}"
                expected_setting="${settings[$((index % 3))]}"
                ;;
            mechanisms)
                methods=(gate/mask_only gate/enforce_only)
                settings=(clean direct indirect)
                expected_method="${methods[$((index / 3))]}"
                expected_setting="${settings[$((index % 3))]}"
                ;;
            oracle)
                methods=(none lexical_guard llm_judge gate/full)
                settings=(target_brand_direct target_brand_indirect near_miss_price_indirect)
                expected_method="${methods[$((index / 3))]}"
                expected_setting="${settings[$((index % 3))]}"
                ;;
        esac

        [[ "$method" == "$expected_method" ]] || fail "$stage[$index] method=$method"
        [[ "$setting" == "$expected_setting" ]] || fail "$stage[$index] setting=$setting"
        expected_defense="${expected_method%%/*}"
        [[ "$defense" == "$expected_defense" ]] || fail "$stage[$index] defense=$defense"
        expected_runtime="none"
        if [[ "$expected_method" == gate/* ]]; then
            expected_runtime="${expected_method#gate/}"
        fi
        [[ "$runtime_mode" == "$expected_runtime" ]] || fail "$stage[$index] runtime=$runtime_mode"
        contains "$result_path" "/results/rebuttal/$stage/$method/$setting.jsonl"
        contains "$summary_path" "/results/rebuttal/$stage/$method/$setting.summary.json"
        contains "$command_line" "-c $checkpoint"
        contains "$command_line" "--type $eval_type"
        contains "$command_line" "--gpu 0"
        contains "$command_line" "--summary_path $summary_path"

        case "$eval_type" in
            clean)
                [[ "$checkpoint" == "$QUERY_CKPT" ]] || fail "clean checkpoint mismatch"
                [[ "$task_ids" == "built-in clean split: std" ]] || fail "clean IDs mismatch"
                ;;
            query_attack)
                [[ "$checkpoint" == "$QUERY_CKPT" ]] || fail "direct checkpoint mismatch"
                [[ "$task_ids" == */sneaker0_test_ids.json ]] || fail "direct IDs mismatch"
                ;;
            observation_attack)
                [[ "$checkpoint" == "$OBS_CKPT" ]] || fail "indirect checkpoint mismatch"
                [[ "$task_ids" == */sneakeri_test_ids.json ]] || fail "indirect IDs mismatch"
                ;;
            *)
                fail "$stage[$index] unknown evaluation type: $eval_type"
                ;;
        esac

        if [[ "$defense" == "gate" ]]; then
            ((gate_rows += 1))
            [[ "$runtime_mode" != "none" ]] || fail "$stage[$index] lacks a GATE mode"
            contains "$command_line" "--gate_runtime_mode $runtime_mode"
            contains "$command_line" "--gate_openai_model test-gate-model"
            contains "$command_line" "--require_goal_parser_success"
            contains "$command_line" "--goal_contract_cache"
        else
            not_contains "$command_line" "--gate_runtime_mode"
            not_contains "$command_line" "--gate_openai_model"
            not_contains "$command_line" "--require_goal_parser_success"
            not_contains "$command_line" "--goal_contract_cache"
        fi

        if [[ "$defense" == "llm_judge" ]]; then
            ((judge_rows += 1))
            contains "$command_line" "--judge_model test-judge-model"
        else
            not_contains "$command_line" "--judge_model"
        fi

        if [[ "$stage" == "oracle" ]]; then
            contains "$command_line" "--stress_test action_oracle"
            if [[ "$setting" == *_direct ]]; then
                contains "$command_line" "--oracle_mode direct_oracle"
            else
                contains "$command_line" "--oracle_mode indirect_oracle"
            fi
            if [[ "$setting" == near_miss_price_* ]]; then
                contains "$command_line" "--oracle_strategy near_miss_price"
            else
                contains "$command_line" "--oracle_strategy target_brand"
            fi
        else
            not_contains "$command_line" "--stress_test"
            not_contains "$command_line" "--oracle_strategy"
        fi

        ((checked += 1))
    done
done

[[ "$checked" -eq 36 ]] || fail "expected 36 rows, checked $checked"
[[ "${#seen_outputs[@]}" -eq 36 ]] || fail "outputs are not unique"
[[ "${#seen_summaries[@]}" -eq 36 ]] || fail "summaries are not unique"
[[ "$gate_rows" -eq 12 ]] || fail "expected 12 GATE rows, got $gate_rows"
[[ "$judge_rows" -eq 6 ]] || fail "expected 6 judge rows, got $judge_rows"

# Model variables are condition-specific: a judge row must not require the
# GATE parser model, and a GATE row must not require the judge model.
judge_only="$({
    env -u OPENAI_API_KEY -u GATE_MODEL \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=9 \
        REBUTTAL_DRY_RUN=true JUDGE_MODEL=test-judge-model bash "$HARNESS"
} 2>&1)" || fail "judge dry run incorrectly required GATE_MODEL or an API key"
not_contains "$judge_only" "DRY_RUN_INVOKED_PYTHON"

gate_only="$({
    env -u OPENAI_API_KEY -u JUDGE_MODEL \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=15 \
        REBUTTAL_DRY_RUN=true GATE_MODEL=test-gate-model bash "$HARNESS"
} 2>&1)" || fail "GATE dry run incorrectly required JUDGE_MODEL or an API key"
not_contains "$gate_only" "DRY_RUN_INVOKED_PYTHON"

# Explicit price snapshots are forwarded only to the selected external-LLM
# role. The cached-input rate remains optional and otherwise uses input price.
priced_judge="$({
    env -u OPENAI_API_KEY -u GATE_MODEL \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=9 \
        REBUTTAL_DRY_RUN=true JUDGE_MODEL=test-judge-model \
        JUDGE_INPUT_USD_PER_MILLION=0.15 \
        JUDGE_CACHED_INPUT_USD_PER_MILLION=0.075 \
        JUDGE_OUTPUT_USD_PER_MILLION=0.6 \
        LLM_PRICING_AS_OF=2026-07-27 \
        LLM_PRICING_SOURCE=official-test-snapshot \
        bash "$HARNESS"
} 2>&1)" || fail "priced judge dry run failed"
priced_judge_command="$(sed -n 's/^Command: //p' <<<"$priced_judge")"
contains "$priced_judge_command" "--judge_input_usd_per_million 0.15"
contains "$priced_judge_command" "--judge_cached_input_usd_per_million 0.075"
contains "$priced_judge_command" "--judge_output_usd_per_million 0.6"
contains "$priced_judge_command" "--llm_pricing_as_of 2026-07-27"
contains "$priced_judge_command" "--llm_pricing_source official-test-snapshot"
not_contains "$priced_judge_command" "--gate_input_usd_per_million"

priced_gate="$({
    env -u OPENAI_API_KEY -u JUDGE_MODEL \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=15 \
        REBUTTAL_DRY_RUN=true GATE_MODEL=test-gate-model \
        GATE_INPUT_USD_PER_MILLION=1.25 \
        GATE_OUTPUT_USD_PER_MILLION=10 \
        LLM_PRICING_AS_OF=paper-snapshot \
        bash "$HARNESS"
} 2>&1)" || fail "priced GATE dry run failed"
priced_gate_command="$(sed -n 's/^Command: //p' <<<"$priced_gate")"
contains "$priced_gate_command" "--gate_input_usd_per_million 1.25"
contains "$priced_gate_command" "--gate_output_usd_per_million 10"
not_contains "$priced_gate_command" "--gate_cached_input_usd_per_million"
not_contains "$priced_gate_command" "--judge_input_usd_per_million"

incomplete_judge_price="$({
    env -u OPENAI_API_KEY -u GATE_MODEL \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=9 \
        REBUTTAL_DRY_RUN=true JUDGE_MODEL=test-judge-model \
        JUDGE_INPUT_USD_PER_MILLION=0.15 \
        bash "$HARNESS"
} 2>&1)" && fail "incomplete judge pricing unexpectedly succeeded"
contains "$incomplete_judge_price" \
    "judge input and output USD-per-million prices must be set together"

provenance_without_price="$({
    env -u OPENAI_API_KEY -u GATE_MODEL \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=9 \
        REBUTTAL_DRY_RUN=true JUDGE_MODEL=test-judge-model \
        LLM_PRICING_AS_OF=2026-07-27 \
        bash "$HARNESS"
} 2>&1)" && fail "pricing provenance without judge prices unexpectedly succeeded"
contains "$provenance_without_price" \
    "LLM pricing provenance requires complete judge prices"

# Relative result/cache overrides must be anchored once, not reinterpreted
# after the job changes into the WebShop evaluator directory.
relative_paths="$({
    env -u OPENAI_API_KEY \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=15 \
        REBUTTAL_DRY_RUN=true GATE_MODEL=test-gate-model \
        RESULTS_ROOT=tmp/rebuttal-results GOAL_CACHE=tmp/shared-goals.json \
        bash "$HARNESS"
} 2>&1)" || fail "relative output/cache dry run failed"
contains "$relative_paths" "Output: $REPO_ROOT/tmp/rebuttal-results/baselines/gate/full/clean.jsonl"
contains "$relative_paths" "Goal cache: $REPO_ROOT/tmp/shared-goals.json"

# Focused cold-cost runs can isolate the GATE cache per resolved row so
# overlapping task goals do not make per-setting spend depend on job order.
row_scoped_cache="$({
    env -u OPENAI_API_KEY \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=16 \
        REBUTTAL_DRY_RUN=true GATE_MODEL=test-gate-model \
        RESULTS_ROOT=tmp/rebuttal-results GOAL_CACHE_SCOPE=row \
        bash "$HARNESS"
} 2>&1)" || fail "row-scoped cache dry run failed"
contains "$row_scoped_cache" "Goal cache scope: row"
contains "$row_scoped_cache" \
    "Goal cache: $REPO_ROOT/tmp/rebuttal-results/baselines/gate/full/direct.goal_contract_cache.json"
row_scoped_command="$(sed -n 's/^Command: //p' <<<"$row_scoped_cache")"
contains "$row_scoped_command" \
    "--goal_contract_cache $REPO_ROOT/tmp/rebuttal-results/baselines/gate/full/direct.goal_contract_cache.json"

invalid_cache_scope="$({
    env -u OPENAI_API_KEY \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=0 \
        REBUTTAL_DRY_RUN=true GOAL_CACHE_SCOPE=invalid bash "$HARNESS"
} 2>&1)" && fail "invalid GOAL_CACHE_SCOPE unexpectedly succeeded"
contains "$invalid_cache_scope" "GOAL_CACHE_SCOPE must be shared or row"

# A requested global GPU must map by position to the corresponding CUDA token.
# This remains correct when Slurm cgroups renumber physical GPUs locally.
physical_gpu_mapping="$({
    env -u OPENAI_API_KEY \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=0 \
        REBUTTAL_DRY_RUN=true PHYSICAL_GPU=6 \
        SLURM_JOB_ID=123 SLURM_JOB_GPUS=0,2,3,6 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash "$HARNESS"
} 2>&1)" || fail "allocated physical GPU selection failed"
contains "$physical_gpu_mapping" "Physical GPU request: 6"
contains "$physical_gpu_mapping" "Slurm global GPUs: 0,2,3,6"
contains "$physical_gpu_mapping" "Slurm CUDA visibility: 0,1,2,3"
contains "$physical_gpu_mapping" "Selected CUDA device: 3"

# Older/differently configured Slurm installations may expose the same global
# allocation through SLURM_STEP_GPUS instead of SLURM_JOB_GPUS.
step_gpu_mapping="$({
    env -u OPENAI_API_KEY -u SLURM_JOB_GPUS \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=0 \
        REBUTTAL_DRY_RUN=true PHYSICAL_GPU=3 \
        SLURM_JOB_ID=123 SLURM_STEP_GPUS=1,3 \
        CUDA_VISIBLE_DEVICES=0,1 bash "$HARNESS"
} 2>&1)" || fail "SLURM_STEP_GPUS fallback failed"
contains "$step_gpu_mapping" "Slurm step GPUs: 1,3"
contains "$step_gpu_mapping" "Slurm GPU allocation source: SLURM_STEP_GPUS"
contains "$step_gpu_mapping" "Selected CUDA device: 1"

# If neither Slurm GPU-ID variable exists, a direct physical numeric CUDA token
# is still verifiable. Renumbered ordinals/UUIDs remain deliberately rejected.
direct_visible_mapping="$({
    env -u OPENAI_API_KEY -u SLURM_JOB_GPUS -u SLURM_STEP_GPUS \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=0 \
        REBUTTAL_DRY_RUN=true PHYSICAL_GPU=3 \
        SLURM_JOB_ID=123 CUDA_VISIBLE_DEVICES=3 bash "$HARNESS"
} 2>&1)" || fail "direct CUDA_VISIBLE_DEVICES fallback failed"
contains "$direct_visible_mapping" \
    "Slurm GPU allocation source: CUDA_VISIBLE_DEVICES-only"
contains "$direct_visible_mapping" "Selected CUDA device: 3"

unverifiable_mapping="$({
    env -u OPENAI_API_KEY -u SLURM_JOB_GPUS -u SLURM_STEP_GPUS \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=0 \
        REBUTTAL_DRY_RUN=true PHYSICAL_GPU=3 \
        SLURM_JOB_ID=123 CUDA_VISIBLE_DEVICES=0 bash "$HARNESS"
} 2>&1)" && fail "unverifiable physical GPU mapping unexpectedly succeeded"
contains "$unverifiable_mapping" "cannot verify physical GPU 3"

if env -u OPENAI_API_KEY \
    REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=0 \
    REBUTTAL_DRY_RUN=true PHYSICAL_GPU=5 \
    SLURM_JOB_ID=123 SLURM_JOB_GPUS=0,2,3,6 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash "$HARNESS" >/dev/null 2>&1; then
    fail "unallocated physical GPU selection unexpectedly succeeded"
fi

invalid_physical_gpu="$({
    env -u OPENAI_API_KEY \
        REBUTTAL_STAGE=baselines SLURM_ARRAY_TASK_ID=0 \
        REBUTTAL_DRY_RUN=true PHYSICAL_GPU=invalid bash "$HARNESS"
} 2>&1)" && fail "invalid PHYSICAL_GPU unexpectedly succeeded"
contains "$invalid_physical_gpu" "PHYSICAL_GPU must be a non-negative integer"

echo "PASS: checked all 36 Slurm dry-run rows without OPENAI_API_KEY or Python invocation"
