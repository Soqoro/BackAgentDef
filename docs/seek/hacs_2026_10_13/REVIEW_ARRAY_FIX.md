# Strict array acknowledgments and evidence-directed review

The supplied v2 development result completed two real pairs, both [1,1], with
two reducer updates. Both victim workers exited successfully. The first Action
review was accepted. The second review twice serialized `observed` as a string
containing JSON instead of an array. The run stopped with
`bounded_review_repair_exhausted`, before anchors and Goal/State reviews.
Preserve this partial result; neither anchors nor qualitative review passed.

The validator now distinguishes `observed_json_array_required` from incorrect
array contents. It does not parse/coerce a string or silently repair evidence.
Canonical JSON comparison also prevents Python's `True == 1` equality from
accepting a boolean as a numerical observation. The existing one-repair budget
is unchanged. Acknowledgment repairs include the expected array as structured
data, its field/type, and an explicit instruction not to quote/stringify it.
The frozen target remains unchanged. Both original replies remain journaled.

Role instructions require prediction assessment to discuss supplied rule IDs,
predicted arm vectors, actual observations, and supplied losses, while retaining
unassessed hypotheses and indistinguishable alternatives. The measured score is
target appearance in a next proposal, not product relevance, search quality,
graph truth, authorization, or a confidence certificate. This is an instruction,
not an automated guarantee of sound prose reasoning; independent review remains
necessary and the matrix stays locked.

Validation: all 329 Seek CPU tests passed; all 52 HACS tests also passed after
extracting the shared repair-packet builder used by the tokenizer diagnostic.
Regression tests exercise strict string/boolean rejection, the exact structured
repair array, a correctly repaired reply, frozen-target preservation and a
second malformed reply exhausting the unchanged budget. `git diff --check`
passed. These fake-backend tests do not establish Qwen schema or reasoning quality.

## CPU cluster checks after syncing

Prepare a new engineering plan/output, preserving v2. No fresh incident or claim
is allocated. A later v3 run repeats development probes rather than pooling the
previous result as new independent evidence.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek -p 'test_hacs.py' -q
python seek_discovery.py hacs-continuation-prepare \
  --result results/seek/hacs_grounding_dev_v1/result.json \
  --parent-plan results/seek/hacs_pilot_v3/incident-0/hacs_active/plan.json \
  --output configs/seek/local/hacs_continuation_dev_v3/plan.json
export SEEK_CONTINUATION_PLAN="$PWD/configs/seek/local/hacs_continuation_dev_v3/plan.json"
export SEEK_CONTINUATION_OUTPUT="$PWD/results/seek/hacs_continuation_dev_v3"
bash seek_hacs_continuation.sh --token-check
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'agent-backdoor-attacks/AgentTuning/WebShop')
from seek.discovery_phase import hacs_runtime as r
from seek.storage import immutable_json
p=json.loads(Path('configs/seek/local/hacs_continuation_dev_v3/plan.json').read_text())
old=Path('results/seek/hacs_continuation_dev_v2/requests/02-0.json')
request=json.loads(old.read_text())
packet=request['key']['packet'];role=request['key']['role']
receipts=[]
for label, payload in (
    ('previous_second_review',packet),
    ('its_array_repair',r.review_repair_packet(packet,'observed_json_array_required',p['outcome']['target'])),
):
    counts=r.role_tokens.measure(p['agents'],r.messages(role,payload))
    receipts.append(dict(packet=label,**counts,within_cap=counts['input_tokens']<=p['input_cap']))
result=dict(plan_hash=p['hash'],development_reuse=True,model_calls=0,
            scope='reused packet size diagnostic, not future-packet guarantee',receipts=receipts)
immutable_json('configs/seek/local/hacs_continuation_dev_v3/evidence_token_preflight.json',result)
print(json.dumps(result,indent=2))
if not all(x['within_cap'] for x in receipts):
    raise SystemExit('Evidence review exceeds cap: do not submit; do not raise cap automatically.')
PY
)
```

Inspect all receipts before submission. Existing budgets remain 4 victim
responses + 2 anchors and 4 logical reviews / at most 8 physical defender calls.
After passing review of the CPU checks:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
export SEEK_CONTINUATION_PLAN="$PWD/configs/seek/local/hacs_continuation_dev_v3/plan.json"
export SEEK_CONTINUATION_OUTPUT="$PWD/results/seek/hacs_continuation_dev_v3"
unset CUDA_VISIBLE_DEVICES
sbatch --partition=PA10080q --nodelist=node04 seek_hacs_continuation.sh
)
```

Current GPU availability, all future packet sizes, schema compliance and
qualitative reasoning remain unresolved. Inspect v3's `result.json` and actual
review journals; a success status alone does not unlock the matrix. No local
GPU inference, downloads, training, submission, registry edits or push occurred.
