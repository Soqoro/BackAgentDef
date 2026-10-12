# Single-Goal recovery after confirmed input-cap failure

The user supplied CPU tokenizer counts of 7143, 7644, 8145, 8646 and 8658.
The final two requests exceed the frozen 8192-token cap. The failed continuation
retained three accepted replies (Goal, State, Action), but no final Goal reply.
Preserve that failed result and all five journaled attempts; do not rerun it.

Recovery reads the old plan/result, verifies historical probe bindings, reconstructs
accepted replies against completed journal entries, and checks that the missing
Goal request contains exactly the seven preserved history entries and full paired
evidence. Its messages must match the saved 04-0.json request. No hypothesis,
condition, weakener or measured outcome is rewritten. The new plan uses current
execution source hashes and explicitly binds the failed result hash.

The CPU preparation measures the exact request using the pinned tokenizer in its
existing Python environment (offline, CUDA hidden, no model load). The request
must exceed the old cap but fit 10240 tokens. Preparation asks for review of this
higher input cap and a single defender attempt before writing the sealed plan.
All other defender settings, including the 1024 output-token cap, remain inherited.
The runner will execute only Goal, append at most one reply after the seven
preserved entries, and make zero victim calls and zero registry writes.

All real role-only requests now receive tokenizer preflight before generation.
An over-cap request is saved with its measured count and stops immediately,
without a journaled generation attempt or formatting retry. A recognized worker
input-cap exception also stops retries. Tokenizer failures fail closed. The
model's own cap check remains active; no truncation is introduced. CPU tests use
simulated token counts and backends; real counts and model behavior require the
cluster assets. The prior run's accounting is not retroactively rewritten.

## CPU preparation and explicit budget review

After syncing the patch:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
mkdir -p configs/seek/local/role_goal_recovery_v1 logs/seek
python seek_discovery.py role-recovery-prepare \
  --failed-root results/seek/role_followup_v1/incident-1 \
  --output configs/seek/local/role_goal_recovery_v1/plan.json
)
```

The command prints the measured size, old/new caps, retained-history count and
one-attempt budget, then asks for approval. Type yes after review. This is a
versioned resource-cap exception for an unchanged request, not permission to
execute a new scientific experiment. No manual JSON edits are needed.

## Submit only the missing Goal turn

The previous Qwen-only A100 80 GB node is used below. Availability and access
must be checked on the cluster; neither is established by local CPU tests.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
export SEEK_REPO="$PWD"
export SEEK_ROLE_CHECK_PLAN=configs/seek/local/role_goal_recovery_v1/plan.json
export SEEK_ROLE_CHECK_OUTPUT=results/seek/role_goal_recovery_v1/incident-1
bash seek_role_check.sh --dry-run
sbatch --partition=HPCAIq --nodelist=node14 seek_role_check.sh
)
```

After completion:

```bash
python -m json.tool \
  "$HOME/BackAgentDef/results/seek/role_goal_recovery_v1/incident-1/result.json"
```

Prerequisites: intact failed continuation plan/result/requests/journal, original
probe evidence and source snapshots, pinned local tokenizer/model files and Python,
review approval, GPU resources, and review of the final real-model reply. A second
failure is retained; do not replay accepted turns or automatically raise caps.
Completion still does not establish strong evidence use, method superiority,
semantic confirmation or malicious training. Any subsequent experiment remains
separately review-gated.
