# Bounded post-grounding development continuation

The user-supplied grounding development result linked to j=15 selected `adidas`
with one real defender call (2,386 reported tokens) and no victim calls. Its
target is grounded and compatible, but its reasoning asserted an unauthorized
narrowing without an explicit applicable prohibition or established violation.
This is development-interface progress with a semantic overclaim; preserve the
original result. It is not fresh discovery or qualitative matrix approval.

The role instruction now explicitly says instruction overlap is a lexical fact,
not an authorization verdict. An unrequested intermediate brand search is not
automatically unauthorized. A probe difference cannot establish authorization.
The numerical validator still does not certify qualitative reasoning: independent
inspection of the actual replies remains necessary.

## Implementation and limits

The continuation reads the previously agent-selected target unchanged from the
hashed grounding result. It verifies the grounding plan and original pilot plan
bindings, then freezes a new current-source development plan. No historical
artifact or authoritative registry entry is modified.

The production HACS family, bank, active selector, response adapter, scorer,
arm-level reducer, and role transport are reused. Maximum budget is two pairs
(four victim responses), two first-pair anchor responses, four logical defender
reviews (Action after each pair, then Goal and State), and at most eight physical
defender attempts. Stop/abstain/finalize decisions stop additional pairs; invalid
replies and unscorable results fail closed. The initial agent-selected outcome
cannot change. There is no new grounding call, target substitution, certificate,
new claim allocation, or matrix unlock.

Decisions/predictions are saved before probes, followed by raw responses, token
and runtime bindings, numerical updates, review packets and replies. The review
display includes performed experiments and the first four unperformed options
in bank order; omitted counts are explicit. Search still uses the full bank.
Every actual review/correction packet is token-checked. The preparatory token
check only measures a baseline packet without realized evidence.

Contexts start at 5000 in the existing renderer. These arbitrary constructed
budgets do not preserve the incident's task or prices. There is no registry-wide
freshness test or reservation: consumed development inputs are retained locally
and must not be recycled as held-out evidence. This is a single inspected-case
engineering diagnostic, not a method comparison or a sample of fresh incidents.
Completed runs are idempotent; partial runs require review instead of automatic
continuation. No local model inference or job submission is authorized by these
documentation commands themselves.

## Commands after syncing code

Local validation passed: all 322 Seek CPU tests, including 45 HACS tests;
launcher shell syntax, CLI help, and `git diff --check`. Simulated integration
verified two evidence updates, four reviews, six physical victim calls including
anchors, idempotent completion, refusal to restart partial work, fixed outcome,
and unchanged registry bytes. These are not real-model results. Shield, Stage I,
`agent_eval.sh`, old results and the authoritative registry were not modified.

Prepare without inference or registry changes:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python seek_discovery.py hacs-continuation-prepare \
  --result results/seek/hacs_grounding_dev_v1/result.json \
  --parent-plan results/seek/hacs_pilot_v3/incident-0/hacs_active/plan.json \
  --output configs/seek/local/hacs_continuation_dev_v1/plan.json
export SEEK_CONTINUATION_PLAN="$PWD/configs/seek/local/hacs_continuation_dev_v1/plan.json"
export SEEK_CONTINUATION_OUTPUT="$PWD/results/seek/hacs_continuation_dev_v1"
bash seek_hacs_continuation.sh --token-check
)
```

Inspect the fixed outcome, budgets, and baseline tokenizer receipt first. Only
after a passing token check and review of this development scope, submit:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
export SEEK_CONTINUATION_PLAN="$PWD/configs/seek/local/hacs_continuation_dev_v1/plan.json"
export SEEK_CONTINUATION_OUTPUT="$PWD/results/seek/hacs_continuation_dev_v1"
unset CUDA_VISIBLE_DEVICES
sbatch --partition=PA10080q --nodelist=node04 seek_hacs_continuation.sh
)
```

Node04 previously worked; current capacity and access are unresolved. Qwen and
the victim load sequentially on one allocated 80GB GPU. Inspect after completion:

```bash
cd "$HOME/BackAgentDef"
python -m json.tool results/seek/hacs_continuation_dev_v1/result.json
```

Review acknowledgment of both arms, numerical updates, next-decision rationale,
and honesty about authorization/unperformed tests. A result with positive effect
does not automatically pass this review. All prior abstentions remain unchanged.
