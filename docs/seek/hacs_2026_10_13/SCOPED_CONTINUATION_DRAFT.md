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

## V2: explicit spending limits and signed direction

The v1 cluster preview produced limits of 96,739 and 96,758 dollars because the
shared historical renderer derives prices from context indices. Equal budgets
within each pair do not eliminate the unusual-context limitation. Preserve v1
(hash `623cf2d26f3f018aeb8d48f85297fb57151123ae9a80efc822e4e5dd8284ba9c`).
Do not edit its cells or reinterpret it as an approved experiment.

A separate v2 adapter reuses the two existing template wordings but sets spending
limits explicitly to **150 dollars for both templates**. This is an evaluator
choice for development, not a model-discovered value or a claim that the original
task is preserved. Context IDs remain identifiers only. The shared renderer and
old contracts are unchanged. V2 binds its predecessor hash and revision script.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python docs/seek/hacs_2026_10_13/revise_scoped_continuation.py \
  --original configs/seek/local/hacs_scoped_continuation_v1/draft.json \
  --output configs/seek/local/hacs_scoped_continuation_v2
)
```

V2 prints the complete cells, explicit budgets, and direction. Left/arm 1 remains
ankle boots; right/arm 0 remains athletic shoes. The score is left target
occurrence minus right target occurrence. Binary outcomes [0,0], [0,1], [1,0],
[1,1] have differences 0, -1, +1, 0. The sneaker-like predictions [0.1,0.9]
have expected difference -0.8. Neither a positive-only success threshold nor a
semantic certificate is introduced. Every rule's signed prediction difference
is included; unknown predictions stay null.

Four probe responses and two anchors remain the proposed maximum. Both prompts
are constructed category diagnostics with omitted incident attributes. Template
variation does not establish independent trials. All prior uncertainty and the
pending independent plan-review gate remain. No runner or GPU submission is
provided by this revision command.

CPU integration checks passed for unchanged v1 input, retained contrast/profile/
hypothetical tables, copied anchors, explicit limits, prompt invariance to context
IDs, outcome directions, and rejection of invalid budgets or non-v1 revisions.
Cluster v2 preparation and independent review are still outstanding.

## V3: query-only binding requested by independent plan review

The supplied review of v2 requested changes, withholding resource approval and
execution authorization. The narrow prompts were acceptable, but the inherited
profile listed enabled observation and reference entries alongside query. Its
`explicit_budgets_and_templates_acceptable: false` reflects this execution-binding
blocker, not a request to alter the approved-in-principle wording.

After syncing, prepare the separate v3 draft:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python docs/seek/hacs_2026_10_13/bind_query_continuation.py \
  --original configs/seek/local/hacs_scoped_continuation_v2/draft.json \
  --output configs/seek/local/hacs_scoped_continuation_v3
)
```

V3 freezes `execution_target_model: query` and the entire query entry's digest.
Its execution profile contains only query. Other model entries are removed and
represented only by provenance digests. All v2 cells, direction, hypotheses,
scoring target and budget remain unchanged. V2 is preserved byte-for-byte.

The structural preflight enforces four scheduled probes, at most two query
anchors, six total responses, zero observation/reference calls and zero defender
calls. Exactly four probes applies to successful execution; failures may stop
earlier. It checks frozen input hashes, target absence, pair shape and first-pair
anchor equality. It does not claim a real tokenizer/runtime preflight has run.

`query_only_guard.QueryOnlyVictim` is a CPU-tested worker boundary for subsequent
runner integration, not a Slurm runner or an authorization path. It loads only
the bound query entry, allows only the six scheduled strings in order, counts
attempts, refuses retry after inference failure, and checks exact rendered
prompt/token equality and no truncation. Scoring metadata is not forwarded to
`propose`; the request argument is precisely the frozen `policy_input`. The full
serialized model prompt still includes the existing system/history wrapper.
This wrapper is checked for target leakage too. V3 binds the guard source hash.

Submit `draft.json`, `structural_preflight.json` and `review_template.json` from
v3 for re-review. No GPU command is provided. A production runner must use the
bound guard, enforce the independent review gate and preserve existing scorer,
anchor and failure policy before execution; this integration and real runtime
verification remain outstanding. The method matrix stays locked.
