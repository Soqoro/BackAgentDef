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

### Audit the six-page collection

`audit_observation_source_collection.py` binds the completed collection
`ae9f9c9255d2819875ae4b06c430e8b5a2950cfc3f347f02b6926f1dab5f8f87`
to its frozen plan and saved page files, checks the catalogue digest, and joins
all six pages using the existing brand/price/legal-click eligibility rules.
It accepts `--collection` and `--output`. It performs no environment actions or
model calls. Pagination retains the preceding search identity. Candidates are
never paired across pages; semantic task fit, benign label choice and observation
control authority remain unresolved even when a structural pair exists.

### Bounded rank diagnostic after the six-page negative result

`audit_observation_retrieval_ranks.py --collection DIR --negative REPORT --output FILE`
binds the negative audit `827bb7f03eccc91580475988f257b158ea0b6f05b53b681165a2cc4eaba415d1`
and repeats only its three frozen query strings against the local Lucene index,
with `k=50` (the existing engine's ordinary-search retrieval cap). It retains all
hits and joins explicit catalogue brands, including unresolved identities.
This does not instantiate WebShop, change its retrieval cap, or run a model.
Direct index ranks are not runtime page positions: runtime filtering, prices,
legal clicks and semantic fit still need separate evidence. No automatic further
collection, favorable-query selection, or slot approval occurs. The index stat
inventory must match the collection receipt; this is not content authentication.

### Alternative query proposal after top-50 negatives

`prepare_alternative_observation_queries.py` takes `--collection`, `--rank-report`
and `--output`. It binds the completed negative rank report, checks the original
catalogue hash, and reuses `inventory`'s explicit-brand mixed-group overview.
It proposes at most three distinct catalogue query strings in lexicographic
category/product-category/query order. Groups containing fragrance, cologne,
parfum, perfume, toilette, shirt or tee (case-insensitive substrings) are excluded
to avoid repeating the inspected task families. This transparent heuristic is
not semantic task validation; all selected strings remain subject to inspection.
The output preserves the full group overview and exclusions. It declares up to
three future top-50 index searches, but executes none. No task, price limit,
source-slot approval or model-execution authority is inferred. Category diversity
is reported rather than assumed, and historical split dependencies stay unresolved.

### Catalogue consistency extraction

`extract_alternative_query_products.py --proposal FILE --collection DIR --output FILE`
binds proposal `26c39e126a89642edb315ca1f579283d11dc1d9eb7a70ce7b50075026750c569`
and replays the existing inventory against the hash-bound catalogue. It retains
all eligible targets and the first five eligible non-target ASINs in lexical order
per selected group, with source positions/hashes, names, raw/parsed brands, raw
prices and category/query fields. Omitted non-target counts are explicit. This
sample is not representative. The Hair Masks/Elastics & Ties discrepancy and
broad Body Care query remain manual review questions. No automatic replacement,
semantic certification, runtime affordability claim or search occurs.
