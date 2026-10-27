# Observation opportunities from saved fragrance pages

This additive read-only builder checks both exact pages from fragrance result
`dbf0fb851928bb07837a12afa794068c835aaef77360e4bfa07f2483d3b9a6de`.
It joins each returned product to the original catalogue bytes and checks saved
legal clickability, explicit brand metadata and numeric runtime price below $50.
Exact Adidas is target; unknown, duplicate and conflicting brand identities remain
unresolved. Other explicit brands are non-target candidates, not automatically
semantically suitable or optimal benign choices.

A candidate label contrast must exist within one saved observation. The script
never pairs a product from the benign-query page with one from the Adidas-query
page. Over-budget products cannot supply the benign alternative. If no pair exists,
it reports a missing opportunity and leaves labels/marker previews unset. This is
a useful bounded negative finding, not a reason to invent a click or relax the task.
If a pair exists, it still requires semantic review and an explicit observation
control boundary. The prior instruction-channel approval does not cover this.

No search, click, model call, new marker, stage assignment or registry mutation is
performed. The task remains development reuse; output is not a training corpus.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/observation-opportunity-XXXXXX")
python scripts/seek/prepare_observation_opportunity_drafts.py \
  --collection "$HOME/seek_private/three_way_v1/fragrance-label-check-ocjihG/collection" \
  --output "$OUT/report.json"
printf '\nEvidence: %s\n' "$OUT/report.json"
)
```

The report retains all selected catalogue records and source positions, runtime
records, excluded reasons and saved observations for audit. No title-based brand
inference or product substitution is applied. Bounded absence on these pages does
not prove that other observation tasks cannot supply a valid contrast.
