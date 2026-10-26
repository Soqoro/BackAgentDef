# Bounded prospective source collection — 2026-10-26

The first cluster attempt (`prospective-environment-b9sGXI`) loaded the catalogue
and 12,087 human goals, then failed at `prospective_goal_reset_mismatch` before
the selected task's search. The original check incorrectly compared the rendered
`Instruction: `-prefixed h4 text with the unprefixed stored goal. The revised
collector compares the full server session goal with a pre-reset deep copy, then
checks rendering against the exact search-page prefix. No whitespace or semantic
normalization is applied. Both goal objects, their hashes, expected/actual text
and initial observation are saved in `reset_checks` before either check fails.
Preserve the failed directory and rerun the complete command below in a fresh
directory with a new preflight; the old source-bound preflight is obsolete.

The evaluator-selected development pilot uses the men's eau-de-toilette group
`599a9ca68d57e1b58cc30cc6cb79a4f0c86b6f4044a270ea51d1210b2e1131e5`
and men's active T-shirt group
`c7521ba9026acb04173bdf5dd023edbc2d7ef42a6cb568243240c75bf375c379`.
The parent full catalogue inventory reported by the cluster is
`e9877904d9564647b6ecd5fa64a928004fb61b075b494ff8f61b63eacd20ec8b`.
This is source acquisition, not attack execution or training-data approval.

The additive `collect_prospective_states` function in `seek/collection.py` is the
only new code that steps the environment. The CLI imports WebShop directly;
it never imports the legacy model entry point, creates a victim or defender,
inserts a marker, clicks a product, purchases, or invokes training. Existing
collectors, filters, Shield, Stage I and agent_eval.sh are unchanged.

The environment uses its existing human goals, the full local catalogue and
search index, with construction seed 42. For each group, candidate goals must
belong to an other-brand product and lack literal references to the group's
known brands and declared target product lines. The lowest goal digest is
chosen before any search. This literal exclusion is not semantic validation.
The original instruction, options and spending limit are never overwritten.
Each chosen goal gets exactly one search using the group's original catalogue
query. At most two searches occur. No extra search is attempted on a miss.

The report saves original goal/hash, initial observation/actions, exact search
action, resulting observation/actions, runtime prices and product dictionaries
for visible shortlist products. A `mixed_brand_state_captured` status means both
candidate brand groups are legally clickable on that result page. It does not
mean they satisfy the goal, that either is an approved benign/poison action, or
that a purchase would succeed. These records are development evidence; stage,
policy serialization, semantic neutrality and insertion authority remain unset
or unverified. All model-execution/training approvals remain false.

Preflight hashes the catalogue, attribute inputs, human instructions and Python/
HTML environment sources. It records index file sizes and timestamps, which do
not authenticate the index bytes or their compatibility with the catalogue.
Collection recomputes the preflight and refuses changes. Installed CPU spaCy,
Java/Pyserini and WebShop dependencies are needed; no download or installation is
attempted. Full-catalogue environment loading can require substantial CPU memory.
Dependency errors are saved in failure.json; existing evidence is not overwritten.

After syncing code, run in the existing `webshop_torchfix` cluster environment:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/prospective-environment-XXXXXX")
export OUT
INVENTORY="$HOME/seek_private/three_way_v1/prospective-source-shortlist-pkOEa2/report.json"

python scripts/seek/collect_prospective_sources.py preflight \
  --inventory "$INVENTORY" --output "$OUT/preflight.json"

python - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['OUT']) / 'preflight.json').read_text())
assert r['status'] == 'assets_present_dependencies_unverified', r['missing_prerequisites']
assert r['inventory_hash'] == 'e9877904d9564647b6ecd5fa64a928004fb61b075b494ff8f61b63eacd20ec8b'
assert r['maximum_search_actions'] == 2
print('Asset gate passed; environment dependencies remain to be tested.')
PY

python -u scripts/seek/collect_prospective_sources.py collect \
  --inventory "$INVENTORY" --preflight "$OUT/preflight.json" \
  --output "$OUT/collection" >"$OUT/run.log" 2>&1 \
  || { tail -n 60 "$OUT/run.log"; printf '\nEvidence: %s\n' "$OUT"; exit 1; }
tail -n 60 "$OUT/run.log"
printf '\nEvidence: %s\n' "$OUT"
)
```

Review the two actual states before any source-slot construction. Missing goals
or missing brand opportunities are results, not authorization to replace tasks
or inject catalogue products into the result page. No clean provenance, split
independence, matched controls or runtime model verification is inferred.

Local validation: 13 CPU fixture tests across source inventory and collection;
no real environment, policy inference, GPU, training, downloads or job submission
was run locally. Real cluster collection is still pending.
