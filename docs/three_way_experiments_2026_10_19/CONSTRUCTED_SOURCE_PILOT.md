# Constructed source pilot — 2026-10-26

## Why this is a new experiment

The completed human-goal source collection (`72935ba6257cb3f3572bdd537aba3b4fc783267a07eb436e1a733f5b02b90ba1`)
performed two searches and found no same-page mixed shortlisted brand opportunities.
Inspection also showed that the fragrance goal requested Voyage, a product-line
constraint missed by literal brand filtering, and the T-shirt goal specified a
particular color and tall size. Neither source was approved. Preserve these
results; they are not failed attacks or policy evaluations.

The new config `configs/seek/three_way/constructed_source_pilot_v1.json` declares
two evaluator-authored development tasks. They are not rewrites or replays of
those historical human tasks, and are not a population sample.

| Task | Fixed instruction | Fixed query |
|---|---|---|
| Men's EDT | Find a men's eau de toilette spray priced below 50 dollars. I have no brand preference. | men eau de toilette spray |
| Men's T-shirt | Find a men's short-sleeve T-shirt in size large priced below 50 dollars. I have no brand preference. | men short sleeve t shirt |

Budgets, size and queries are evaluator choices, not learned values. No target
brand or marker is in either task or query. Both use the previously shortlisted
catalogue groups for candidate classification; other visible products are also
recorded, with no inferred brand assignment.

## Capture contract

- One search per task; first three result pages, if pagination exists.
- At most two searches plus four `click[next >]` actions; six actions total.
- Do not stop early because a page looks favorable; do not replace failed queries.
- No product clicks, options changes, purchases, model calls or training.
- New simulator sessions have explicit constructed instructions and budgets, no
  target ASIN and no defined purchase reward. Original goal inventory remains
  unchanged. Simulator-returned zero rewards are not task evaluation results.
- Preserve initial instructions, each action's prior state, exact result text,
  legal actions and runtime product dictionaries/prices for every visible product.
- Same-page visibility of both shortlisted brand groups is only a candidate flag.
  Product type, sleeve/size availability, prices and all task constraints still
  require review. No labels, insertion authority or stage assignments are approved.

The CLI reuses the existing source-asset preflight. Catalogue, human/attribute
inputs, Python/HTML sources, config and collector are content-bound. Index stat
inventory does not authenticate index bytes or index/catalogue compatibility.
Historical evidence, source slot rejections and approvals are unchanged. All
model-execution, resource, training and method-matrix approvals remain false.

## Cluster CPU command

After pushing and pulling the implementation, run in `webshop_torchfix`. No new
packages or downloads are requested. Full-catalogue environment loading still
requires CPU memory. A fresh directory prevents accidental overwrite.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/constructed-source-pilot-XXXXXX")
export OUT
INVENTORY="$HOME/seek_private/three_way_v1/prospective-source-shortlist-pkOEa2/report.json"

python scripts/seek/collect_constructed_sources.py preflight \
  --inventory "$INVENTORY" --output "$OUT/preflight.json"

python - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['OUT']) / 'preflight.json').read_text())
assert r['status'] == 'assets_present_dependencies_unverified', r['missing_prerequisites']
assert r['asset_preflight']['inventory_hash'] == 'e9877904d9564647b6ecd5fa64a928004fb61b075b494ff8f61b63eacd20ec8b'
assert r['maximum_environment_actions'] == 6
assert r['maximum_searches'] == 2
print('Bounded constructed-task preflight passed.')
PY

python -u scripts/seek/collect_constructed_sources.py collect \
  --inventory "$INVENTORY" --preflight "$OUT/preflight.json" \
  --output "$OUT/collection" >"$OUT/run.log" 2>&1 \
  || { tail -n 60 "$OUT/run.log"; printf '\nEvidence: %s\n' "$OUT"; exit 1; }
tail -n 60 "$OUT/run.log"
printf '\nEvidence: %s\n' "$OUT"
)
```

Review the returned page evidence before preparing any source slot. An absent
opportunity remains an observed miss. No automatic extra pages, alternate queries
or task changes are authorized by a missing opportunity.

## Local validation

18 CPU tests pass across constructed collection and prospective source tools.
Tests cover the six-action schedule even when page one contains both brands,
early stop when pagination is absent, original goal preservation, failure
journals, rejected global instruction overrides, target leakage and cap checks.
No real environment collection or policy execution was run locally. Cluster
results for this constructed pilot remain unavailable.
