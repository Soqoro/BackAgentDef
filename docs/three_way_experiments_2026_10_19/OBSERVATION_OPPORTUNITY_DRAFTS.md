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

## Separate prospective source schedule

The negative audit `2daee46579ba281d77848599d239e28564a54d7d2885a24800d7da6d633c3c52`
is retained. A separate development schedule freezes three neutral query strings:
`eau de toilette for men`, `mens cologne spray`, and `men fragrance spray`.
Each uses a fresh session with the unchanged men's EDT spray below-$50 task, its
first page and at most one next page. Maximum: three searches plus three pagination
clicks. No product click, purchase, retry, query substitution or first-hit stop.
All pages must be retained, including failures to supply a contrast.

This is prospective with respect to new collection results only. The task and
query design are post-inspection development reuse, not independent held-out
sampling. Explicit brand, legal click and price checks are structural filters;
both candidate products still need semantic compatibility and label review.
Observation control remains unresolved. The initial-instruction query approval
cannot be relabelled as observation-field permission.

The preparation command hashes existing assets and writes a frozen plan. It does
not instantiate WebShop or implement/execute a collector. All model approvals
remain false, and no training records or observation labels are created.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/observation-source-plan-XXXXXX")
python scripts/seek/prepare_observation_source_plan.py \
  --negative "$HOME/seek_private/three_way_v1/observation-opportunity-De9GLI/report.json" \
  --collection "$HOME/seek_private/three_way_v1/fragrance-label-check-ocjihG/collection" \
  --output "$OUT/plan.json"
printf '\nFrozen plan: %s\n' "$OUT/plan.json"
)
```

## Execute the frozen model-free collection

`collect_observation_sources.py` accepts only plan
`161568d6c51fc04177fe00339e0f89ac5179312256e86e164f686c37b76bafb9`.
It does not change that plan's historical `collector_implemented=false` field.
The new collector is separately source-bound in its runtime evidence.

Before environment loading it rechecks asset bytes, schedule and design file.
It records current environment sources and index statistics (not authenticated
index contents), creates an exclusive output directory, and retains the exact
plan. Each fresh query session has a saved reset state. Every action has a durable
pre-action record and attempt journal; all returned pages are retained. Maximum
six actions: exactly the three scheduled searches, with one optional next-page
click per query. No favorable-result stopping or product selection occurs.
Missing pagination ends that query only; exceptions stop the whole collection.
Failure evidence counts attempts and completed pages. There is no resume/retry
option. A rerun would be a new deliberate collection, not a continuation.

Use the existing `webshop_torchfix` environment:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/observation-source-collection-XXXXXX")
python scripts/seek/collect_observation_sources.py \
  --plan "$HOME/seek_private/three_way_v1/observation-source-plan-aE8b30/plan.json" \
  --output "$OUT/collection" >"$OUT/run.log" 2>&1 \
  || { tail -n 60 "$OUT/run.log"; exit 1; }
tail -n 60 "$OUT/run.log"
printf '\nEvidence: %s\n' "$OUT/collection"
)
```

Saved runtime products/prices/clickables are evidence, not automatically reviewed
brand or semantic compatibility. No observation edit or click labels are assigned.
All model-execution/training approvals remain false; only CPU environment actions
in this frozen collection are performed.
