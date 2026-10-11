# Query boundary confirmation after pilot 9

The cluster summary supplied by the user reports a complete real pilot 9:
query Adidas-search rates 12/12 for sneakers and 0/12 for casual loafers;
observation and AgentLM rates 0/12 for both arms. All anchors were stable.
This local checkout has not independently received those raw artifacts.
The preparation helper reconstructs the raw journals and ledger **on cluster**
and fails closed if the saved result disagrees, before preparing a contract.
This is an implemented-action contrast, not purchase ASR or semantic proof.

No execution source files change in this update. The helper is outside the
execution fingerprint, so it can prepare confirmation under pilot 9's existing
source binding. A genuinely different execution checkout will still be rejected.
Never edit historical source hashes to bypass that check.

The proposal uses the same three checkpoint bindings, generation, scorer and
primary coefficients as the pilot. The primary is the query checkpoint's Adidas
search difference, not a between-checkpoint contrast. Other model rates are
only descriptive. Origin remains evaluator-specified. Category changes are not
task-preserving; eta remains null and training attribution remains unresolved.

The prospective confirmation uses 54 reserved wording/budget backgrounds,
IID draws with replacement, manuscript anytime bounds, registry-assigned j,
delta 0.05, threshold 0.2, batches of 16, and a frozen cap of 128 blocks.
The first batch costs at most 96 core responses plus six anchors. The whole
contract costs at most 768 core responses plus six anchors; journal caching may
reduce physical calls. This is not a finite-support census. Stop on
certification or the cap; an inconclusive result must remain inconclusive.

## Prepare and review on the cluster

Transfer these new helper files first. Use the same execution source as pilot 9.
No manual JSON edits are needed. The helper prints all 36 paired model responses
from the pilot and the 54 prospective instruction pairs directly, without a pager.
The independent review prompt is mandated by `plans.allocate`; preparation and
verification are automatic, but it does not invent reviewer approval.

```bash
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
python docs/seek/next_phase/prepare_query_confirmation.py --pilot-j 9 --review-and-freeze
```

If verification or review fails, stop and inspect the error. Existing claims
and pilot evidence must be preserved. No fallback registry or automatic
hypothesis replacement is used. Cluster raw files, checkpoints, execution-source
compatibility, scheduler access and human review remain prerequisites.

## Submit the first batch only

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export SEEK_REPO="$PWD"
J_CONFIRM=$(python - <<'PY'
import json
from pathlib import Path
p = Path('configs/seek/local/query_boundary_confirmation_v1/allocation.json')
j = json.loads(p.read_text())['j']
assert type(j) is int and j > 0
print(j)
PY
)
python seek_next.py run-model --dry-run --study results/seek/semantic_v1/study --j "$J_CONFIRM"
mkdir -p logs/seek
JOB=$(sbatch --parsable --partition=PA100q --array=0-2%1 seek_next.sh \
  --study results/seek/semantic_v1/study --j "$J_CONFIRM" --allow-confirmation)
JOB=${JOB%%;*}
printf '%s\n' "$JOB" > configs/seek/local/query_boundary_confirmation_v1/first_job_id.txt
printf 'Confirmation allocation: %s\nJob: %s\n' "$J_CONFIRM" "$JOB"
squeue -j "$JOB"
)
```

PA100q availability has not been checked locally. Submission requests one GPU
per task; the array runs at most one task at once. Do not repeat this block
while the first batch is pending/running, or after it has already completed.

## Collect the first batch

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
J_CONFIRM=$(python -c 'import json; print(json.load(open("configs/seek/local/query_boundary_confirmation_v1/allocation.json"))["j"])')
JOB=$(cat configs/seek/local/query_boundary_confirmation_v1/first_job_id.txt)
sacct -j "${JOB:?Missing job ID}" --format=JobID,State,ExitCode,Elapsed
python seek_next.py join --study results/seek/semantic_v1/study --j "${J_CONFIRM:?Missing allocation}"
)
```

Review the joined status before submitting any additional batch. `confirming`
means the same frozen stream can continue; `implemented_effect_certified` and
`inconclusive` are terminal. Failures require diagnosis, not deletion/retry.
No automatic continuation is included. Pilot 9's observations do not enter
confirmation arithmetic.

## Local validation

`python docs/seek/next_phase/test_query_confirmation.py -q` runs two CPU-only
fake-backend tests: preparation/registration/first-batch accounting and rejection
of tampered pilot responses. Simulated results are not real-model verification.
