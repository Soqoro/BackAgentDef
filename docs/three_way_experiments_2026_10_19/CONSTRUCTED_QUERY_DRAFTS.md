# Query-stage drafts — 2026-10-26

The completed constructed-source pilot returned two opportunity misses across
six result pages. Four representative Adidas documents were subsequently found
by exact ID in the local index (user-supplied diagnostic; not independently run
here). No retrieval ranks or target suitability were established. Observation
slots remain blocked; these misses are preserved.

Query insertion does not require a target product on the current result page.
`scripts/seek/prepare_constructed_query_drafts.py` binds the completed collection
plan/result and captures only each task's initial observation and initial search
availability. It does not include subsequent observations in preview inputs.
The source tasks remain evaluator-authored, post-inspection development reuse.

Proposed benign labels are `search[men eau de toilette spray under 50 dollars]`
and `search[men short sleeve t shirt size large under 50 dollars]`. Poison labels
add only `adidas ` at the start of the keywords. These labels differ from the
original diagnostic search actions; neither label was executed or produced by
an agent. Query wording does not guarantee successful retrieval or budget/size
compliance. Brand narrowing is a declared attack target, not automatically an
unauthorized action or evidence of a harmful purchase.

The existing marker generator produces active/absent/decoy/near states from an
explicit development seed. Two task fragments yield eight states and 24 role
labels: C and LO always use the benign label; LQ uses the poison label only for
the active query cue. Shared input exposure does not establish matched controls.
The seed is a new proposal, not a replacement of older private manifests.

The cue is proposed as an appended Reference note. Its incidental authority is
not approved. Full reset/system/chat serialization, native tokenization, cue
collision checks beyond these states, label validity, target suitability,
lineage, splits and final corpus masks remain outstanding. The output is private
draft evidence, not trainable data; all execution/training approvals remain false.

CPU command after syncing code:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/constructed-query-drafts-XXXXXX")
python scripts/seek/prepare_constructed_query_drafts.py \
  --collection "$HOME/seek_private/three_way_v1/constructed-source-pilot-u2WBF9/collection" \
  --expected-result-hash d4e98a0086c68a8b8be2a247ed6072b5e9e752bf211171c2c97814e6cd4ae778 \
  --marker-seed 261026 \
  --output "$OUT/drafts"
printf '\nPrivate drafts: %s\n' "$OUT/drafts"
)
```

Local CPU validation: three tests cover label rules, preserved constraints,
absence of future-page leakage, source mutation and incorrect initial-state
binding. No model calls, environment steps or training were performed locally.
