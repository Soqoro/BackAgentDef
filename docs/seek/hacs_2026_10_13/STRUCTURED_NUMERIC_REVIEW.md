# Structured phase-separated review: v6 engineering check

This supersedes the prior saved-review submission instructions. Pause GPU review
iterations while reviewing the schema and tokenizer checks below.

## Evidence motivating the change

Job 1097658 completed on NA100q in 6:06. Saved-review v5 reported four defender
calls, 17,411 tokens, zero victim calls, no failure and no registry mutations.
Result hash: `82e2cd6b7d593f430cbe95d56af5ff5a810c08108af69ec663566bbf1fa87af3`.
It remains an engineering completion with a failed non-independent qualitative
assessment. Reviews confused past low losses with predictions for a different
proposed test and failed to describe the four hypothetical outcomes correctly.
No existing review, result, claim or registry entry is changed.

## Contract

The saved-review packet now has two named sections:

- `observed_evidence`: performed probe ID, performed experiment ID, actual
  precommitted predictions and measured losses for displayed rules.
- `proposed_test`: prior agent-selected experiment, operands, `executed:false`,
  next-test predictions and four hypothetical Brier-loss outcomes.

The review adds `numeric_evidence` with `rule_ids`, `observed_predictions`,
`observed_losses`, `proposed_predictions`, and `hypothetical_losses`. The agent
chooses exactly two distinct displayed rule IDs in `hypothesis_ids`; every
numeric array must refer to those same IDs in that order. The validator compares
canonical JSON against the two packet sections, rejecting swapped or invented
numbers, boolean substitution, strings containing JSON, and absent/unknown IDs.
The hypothetical-loss array has two rows, one per rule, each in outcome order
[0,0], [0,1], [1,0], [1,1]. Unknown values remain null.

The proposed test is the one bound from the prior completed agent review. A new
test requires a separate contract; the validator does not choose a replacement.
The model may still stop or abstain while acknowledging evidence. There is no
required winning hypothesis, effect or favored scientific conclusion.

Numerical checking establishes faithful transcription only. Prose may contradict
correct structured fields. Result metadata retains `qualitative_assessment:
unresolved`, independence false, matrix locked and no scientific confirmation.
Independent qualitative review remains necessary. One factual/schema repair is
allowed; incomplete or truncated replies still fail immediately.

To retain token headroom, exploratory weight values are omitted from this
saved-review display with an explicit field/digest record. The full parent
artifacts and controller weights remain unchanged. Actual observed predictions,
losses, acknowledgment arrays, rule definitions, probe inputs and hypothetical
outcome rows remain visible. Full graph/initial grounding context remains omitted
as in v5; neither graph validity nor weight ranking is assessed here.

The numerical validator is opt-in through the existing review-call function.
General HACS, Shield and Stage I retain their prior review schema and behavior.
No framework is rebuilt and no victim calls are added.

## CPU and tokenizer commands after syncing

Create a new v6 plan; preserve every earlier version:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek \
  -p 'test_hacs*.py' -q
python docs/seek/hacs_2026_10_13/review_saved_evidence.py prepare \
  --run-root results/seek/hacs_continuation_dev_v3 \
  --prior-review results/seek/hacs_saved_review_v5/result.json \
  --plan configs/seek/local/hacs_saved_review_v6/plan.json
python docs/seek/hacs_2026_10_13/review_saved_evidence.py token-check \
  --plan configs/seek/local/hacs_saved_review_v6/plan.json \
  --output configs/seek/local/hacs_saved_review_v6/token_preflight.json
)
```

There are now **sixteen receipts**: four initial packets plus array-type,
acknowledgment-value and numerical-reference repair packets for each review.
All must fit 4096 input tokens. Output remains 1024 tokens, four logical reviews,
eight maximum defender attempts, zero victim calls. Passing token checks does
not establish output fit or real-model reasoning correctness. Do not submit a
GPU run yet; review the new structure and receipts first. No new model calls,
training, downloads, API calls, sbatch or pushes were performed locally.

## Local validation

All 338 Seek CPU tests passed; the focused HACS suite passed 57 tests. Synthetic
regressions cover phase-specific numerical checks, malformed/string/boolean
values, output-limit rejection, preserved parent artifact bytes, successful
four-review orchestration, one correction per review (eight attempts total),
and immediate truncation failure. A test deliberately leaves unsupported prose
accepted by the numeric validator, demonstrating why qualitative review remains
separate. These tests use fake backends; no real Qwen result is inferred.
`git diff --check` and CLI help checks passed.
