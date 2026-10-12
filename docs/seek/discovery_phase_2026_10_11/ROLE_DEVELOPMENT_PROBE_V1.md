# Bounded development probe after V5

The user-supplied V5 result reports four executable agent-authored requests,
12,610 reported defender tokens, and zero victim calls. The selected request is
sneakers versus trainers, measuring Adidas search insertion. Role agreement does
not establish scientific validity: the recorded weakener repeats the suspicion,
and Goal/State provide little substantive challenge. Preserve the authored
condition and weakener; do not replace them with evaluator-written reasoning.

This extension prepares an explicitly evaluator-selected development probe of
the unchanged final State request approved by Action. It is not a new incident-led
method trial, held-out test, finite-support census, or statistical certificate.
The parent allocation, role plan, role result and authored lineage are hash-bound
and retained in the new plan. Historical source maps are read as historical;
new execution sources are frozen independently. No historical plan is resumed.

The probe reuses methods.cells, the existing response journal, strict action
scorer and panel runner. It contains four pairs (two existing discovery frames
at each of budgets 610 and 627), eight core responses and two uncached repeat
checks of the first pair. Only the query checkpoint is loaded. Maximum physical
responses: 10; defender calls: 0. The plan preserves the parent generation and
model identity. Raw responses, consumed prompts/tokens, runtime bindings and
costs are retained. Truncation, target leakage and source mismatches fail closed.
Unscorable pairs are never discarded; descriptive means are withheld on missing
scores or unstable repeats. Interrupted attempts cannot be automatically rerun.

This does not allocate a claim. Existing registry events remain intact; consumed
prompt/token exposures are appended through the existing registry mechanism so
these calls cannot silently become fresh evidence. Overlap with another stream
blocks execution and requires review, not a new path to bypass the check.

Interpretation, explicitly evaluator-provided rather than agent-authored:
- Equal insertion rates weaken a wording-specific difference on these inputs,
  but do not exclude category-wide effects or ordinary brand preference.
- A descriptive difference motivates further investigation, not a backdoor claim.
- Both terms refer to similar footwear. Neither arm establishes category absence.
- Constructed frames omit original attributes and prices: this is not a
  task-preserving counterfactual replay of the incident.
- No purchase, episode reward, unauthorized-action or malicious-training claim
  follows. No subsequent matrix or confirmation is automatically authorized.

## Prepare and review on the cluster (CPU only)

After syncing code, run the following. No file editing is needed. The review
command prints eight inputs, repeat IDs, model/generation settings and limitations.
The explicit prompt gates the ten-response budget; type yes only after review.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
mkdir -p configs/seek/local/role_probe_v1 logs/seek
python seek_discovery.py role-probe-prepare \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/discovery_phase_v2/method_allocation.json \
  --role-plan configs/seek/local/role_grounding_v5/plan.json \
  --role-result results/seek/role_grounding_v5/incident-1/result.json \
  --output configs/seek/local/role_probe_v1/plan.json
python seek_discovery.py role-probe-review \
  --plan configs/seek/local/role_probe_v1/plan.json \
  --output configs/seek/local/role_probe_v1/review.json
)
```

## Submit only the reviewed probe

This loads the 7B query victim, not Qwen. PA100q access and availability must be
checked on the cluster; this runbook does not assert current free GPUs.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
export SEEK_REPO="$PWD"
export SEEK_STUDY=results/seek/semantic_v1/study
export SEEK_ROLE_PROBE_PLAN=configs/seek/local/role_probe_v1/plan.json
export SEEK_ROLE_PROBE_REVIEW=configs/seek/local/role_probe_v1/review.json
export SEEK_ROLE_PROBE_OUTPUT=results/seek/role_probe_v1/incident-1
bash seek_role_probe.sh --dry-run
sbatch --partition=PA100q seek_role_probe.sh
)
```

After completion:

```bash
cd "$HOME/BackAgentDef"
python -m json.tool results/seek/role_probe_v1/incident-1/result.json
```

Cluster prerequisites: historical allocation and V5 source/result artifacts,
source snapshots for binding checks, query checkpoint weights matching the frozen
identity, compatible existing environment, writable authoritative exposure journal,
approved review receipt, available GPU and user inspection of real output. Local
CPU tests use simulated policies and do not complete this experiment.
