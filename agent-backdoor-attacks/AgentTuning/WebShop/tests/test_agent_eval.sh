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
if grep -Eq '^[[:space:]]*(export[[:space:]]+)?CUDA_VISIBLE_DEVICES=' "$HARNESS"; then
    fail "agent_eval.sh assigns CUDA_VISIBLE_DEVICES"
fi

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

echo "PASS: checked all 36 Slurm dry-run rows without OPENAI_API_KEY or Python invocation"
