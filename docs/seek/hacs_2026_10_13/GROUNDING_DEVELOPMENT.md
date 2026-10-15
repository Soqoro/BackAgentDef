# Grounding inventory development diagnostic

The user-supplied j=15 journal contains two completed Qwen calls: an invalid
`sneakers` target, followed by abstention after the compatibility correction.
The second reply incorrectly states that no other grounded target exists.
Preserve j=15 as an abstention; this diagnostic does not rewrite that assessment
or allocate a replacement scientific claim.

## Changes

Grounding packets now contain a controller-generated lexical inventory. It lists
ASCII alphabetic/hyphen tokens of length 2–40 in original action order, including
rejected entries, exact character spans, a source hash, instruction overlap,
grounding validity, and compatibility with all supported query frames. It retains
the first 64 occurrences and reports omitted occurrences. This is deliberately
not exhaustive multiword phrase enumeration, semantic extraction, or ranking.
All enumerated compatible candidates are retained; none is selected. Agents can
still propose any valid grounded phrase or abstain. No brand-specific rule exists.
Compatibility does not establish authorization, anomaly, or malicious training.

The correction now distinguishes an unaccepted initial proposal (which may be
repaired or abandoned) from a frozen accepted target (which cannot change).
An accepted abstention is never retried to elicit a preferred outcome.

The new `hacs-grounding-prepare` and `hacs-grounding-run` commands reuse the old
incident and pinned defender settings but bind a new plan to current code. They
explicitly label development reuse and parent result/plan hashes. There is no
registry access, claim allocation, victim loader, search, effect certification,
or matrix unlock. Budget: one logical Action call, at most two physical attempts.
Every actual packet, including a correction, gets the existing token preflight.
A completed identical diagnostic is returned without additional calls; a partial
run requires review and is not silently retried.

Local validation: all 319 Seek CPU tests passed, including 42 HACS tests.
The new launcher passed `bash -n`, both CLI help commands passed, and
`git diff --check` passed. Fake-backend tests checked source-order spans,
compatibility/rejection/negation, enumeration limits, bounded attempts, resume
behavior, and unchanged parent-result and registry bytes. No real model run or
cluster token check has been performed for this change.

## Cluster commands (after syncing this correction)

Do not collect another incident. First prepare and inspect the mechanical
inventory and CPU tokenizer receipt:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek -p 'test_hacs.py' -q
python seek_discovery.py hacs-grounding-prepare \
  --result results/seek/hacs_pilot_v3/incident-0/hacs_active/result.json \
  --output configs/seek/local/hacs_grounding_dev_v1/plan.json
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'agent-backdoor-attacks/AgentTuning/WebShop')
from seek.discovery_phase.hacs_grounding import packet
p=json.loads(Path('configs/seek/local/hacs_grounding_dev_v1/plan.json').read_text())
print(json.dumps(packet(p)['outcome_inventory'],indent=2))
PY
export SEEK_GROUNDING_PLAN="$PWD/configs/seek/local/hacs_grounding_dev_v1/plan.json"
export SEEK_GROUNDING_OUTPUT="$PWD/results/seek/hacs_grounding_dev_v1"
bash seek_hacs_grounding.sh --token-check
)
```

Only after inspecting the inventory and a `within_cap: true` receipt, submit this
bounded development diagnostic. Submission authorizes the two-attempt maximum;
it is not independent scientific review. Node04 previously worked, but current
availability and access are not verified locally.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
export SEEK_GROUNDING_PLAN="$PWD/configs/seek/local/hacs_grounding_dev_v1/plan.json"
export SEEK_GROUNDING_OUTPUT="$PWD/results/seek/hacs_grounding_dev_v1"
unset CUDA_VISIBLE_DEVICES
sbatch --partition=PA10080q --nodelist=node04 seek_hacs_grounding.sh
)
```

After completion:

```bash
cd "$HOME/BackAgentDef"
python -m json.tool results/seek/hacs_grounding_dev_v1/result.json
```

Review both replies/correction if present, not only whether a target was selected.
A successful response here establishes at most development-interface usability,
not fresh discovery, search performance, semantic recovery, or matrix approval.
The old source manifests remain unchanged; new plans bind the updated source.
