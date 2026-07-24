#!/bin/bash

set -euo pipefail

TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TEST_DIR/../../../.." && pwd)"
HARNESS="$REPO_ROOT/choice_integrity_eval.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

contains() {
    local text="$1"
    local expected="$2"
    [[ "$text" == *"$expected"* ]] || fail "missing expected text: $expected"
}

# A dry-run entrypoint must resolve the matrix without importing the evaluator.
python() {
    echo "DRY_RUN_INVOKED_PYTHON" >&2
    return 97
}
export -f python

bash -n "$HARNESS"
grep -q '^#SBATCH --array=0-17%2$' "$HARNESS" \
    || fail "the launcher does not embed the complete 18-cell array"
grep -q '^#SBATCH --gres=gpu:1$' "$HARNESS" \
    || fail "each cell must request exactly one GPU"

methods=(
    undefended
    gate
    state_aware_verifier
    gate_ci
    gate_ci_no_ledger
    gate_ci_no_dominance
)
conditions=(clean direct indirect)
declare -A seen_cells=()

for ((index = 0; index < 18; index++)); do
    output="$({
        env -u OPENAI_API_KEY \
            REPO_ROOT="$REPO_ROOT" \
            DRY_RUN=1 \
            MATRIX_INDEX="$index" \
            RUN_ID=choice-integrity-shell-test \
            bash "$HARNESS"
    } 2>&1)" || fail "matrix row $index dry run failed"

    [[ "$output" != *"DRY_RUN_INVOKED_PYTHON"* ]] \
        || fail "matrix row $index invoked Python during dry run"

    expected_method="${methods[$((index / 3))]}"
    expected_condition="${conditions[$((index % 3))]}"
    contains "$output" "Array index: $index"
    contains "$output" "Method: $expected_method"
    contains "$output" "Condition: $expected_condition"
    contains "$output" "Checkpoint: <MODEL_CHECKPOINT-required>"
    contains "$output" "DRY_RUN=1; filesystem, environment, import, and GPU preflights are skipped."

    cell_dir="$(sed -n 's/^Cell directory: //p' <<<"$output")"
    [[ -n "$cell_dir" ]] || fail "matrix row $index has no cell directory"
    contains "$cell_dir" \
        "/runs/choice-integrity-shell-test/cells/$expected_method/$expected_condition/seed_42"
    [[ -z "${seen_cells[$cell_dir]+present}" ]] \
        || fail "duplicate cell directory: $cell_dir"
    seen_cells["$cell_dir"]=1

    run_command="$(sed -n '/^Run command:$/,+1p' <<<"$output")"
    contains "$run_command" "choice_integrity_eval.py run"
    contains "$run_command" "--method $expected_method"
    contains "$run_command" "--condition $expected_condition"
    [[ "$run_command" != *"choice_integrity_eval.py build"* ]] \
        || fail "evaluation row $index tries to rebuild the frozen benchmark"
done

[[ "${#seen_cells[@]}" -eq 18 ]] || fail "expected 18 unique cells"

if REPO_ROOT="$REPO_ROOT" DRY_RUN=1 MATRIX_INDEX=18 \
    bash "$HARNESS" >/dev/null 2>&1; then
    fail "out-of-range matrix index unexpectedly succeeded"
fi

echo "PASS: checked all 18 choice-integrity Slurm rows without Python or an API key"
