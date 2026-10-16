# Ranking qualification and scoped continuation draft

The cluster comparison shows that the retained role choice, ankle boots versus
athletic shoes, scores 0.149951 after probe-0 against the top 0.159477 (about 6%
lower). After probe-1 it scores 0.152687 against 0.155735 (about 2% lower).
Ranks 2–5 after probe-1 have exactly the same utility; rank 5 reflects ID tie
breaking within that group. These are heuristic utility gaps, not significance
levels or estimates of scientific superiority. The role choice has a defensible
narrow rationale. The negative qualitative verdict remains justified by the
incorrect losses, mechanism overclaims and inadequate treatment of alternatives,
not by its ordinal ranking alone.

The reviewer-suggested boots/backpacks test separates sneaker-like, footwear and
constant-high predictions but scores lower under the existing full-family
objective. It cannot separate constant-high from merchandise. It must not replace
the original agent choice silently. All five null-prediction rules remain
unassessed in the printed candidates.

## Prepare for review, without execution

After syncing, run:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python docs/seek/hacs_2026_10_13/prepare_scoped_continuation.py \
  --parent results/seek/hacs_continuation_dev_v3 \
  --reference results/seek/hacs_reference_review_v1 \
  --comparison results/seek/audits/hacs_reference_selection_v1/comparison.json \
  --output configs/seek/local/hacs_scoped_continuation_v1
)
```

This creates immutable `draft.json` and a pending `review_template.json`.
The exact selected experiment comes from the saved role replies; disagreement
or a missing choice blocks preparation. The builder reuses the existing library,
renderer, target and profile. No target, contrast, confidence rule or hypothesis
is selected to obtain a preferred outcome.

The draft proposes two paired contexts (four responses) and repetition of the
first pair (two anchors), with no defender calls or automatic retries. It carries
all rule predictions and four hypothetical binary outcomes with per-rule mean
squared losses. Zero differential response is a reportable outcome. Missing or
malformed actions and unstable anchors stop interpretation; they are not excluded.

Both contexts use the next indices after the parent run. They are not thereby
fresh, independent or held out. The renderer's category changes, generic shopping
frames and generated prices must be examined by the reviewer. The original
incident attributes and price limits are omitted. Nothing in preparation checks
new consumed tokenizer identities, real response validity or runtime readiness.
These remain prerequisites before execution.

For the retained contrast, footwear, constant-high and merchandise predict high
in both arms; sneaker-like predicts low/high. All outcomes leave some competing
rules unresolved. The scope is descriptive target occurrence in constructed
next-action prompts, not semantic activation, task success, authorization,
training causation, unique mechanism identification or method superiority.

The reviewer must separately judge selection provenance, exact rendered inputs,
hypothetical outcomes and residual ties, budget and failure policy. The template
starts pending and grants no approval. There is deliberately no submission or
execution command in this preparer. A negative review remains negative; this new
plan cannot retroactively pass prior role reasoning or unlock the method matrix.

Local synthetic integration verifies retained selection, parent/plan/comparison
binding, deterministic draft preparation, four probe cells and the six-response
cap. Real cluster artifacts and independent plan review remain unresolved.
