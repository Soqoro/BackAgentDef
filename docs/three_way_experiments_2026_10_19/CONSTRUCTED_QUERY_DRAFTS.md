# Query-stage drafts — 2026-10-26

## Explicit instruction-channel revision

Review of cluster draft `b9f4be16bb93b7c0bc50a1d7887925c44c15ac55bfbca48233850fd002502fbc`
found that the proposed note followed the Search affordance. That is not an
explicit instruction-field insertion. Preserve v1. The additive
`revise_query_instruction_placement.py` produces v2 with the same markers and
labels, moving only the proposed note to the end of the instruction field before
` [SEP] Search`. It verifies the parent/marker seals, exact captured structure,
old placement and label rules. Each preview records the inserted character span.
Removing that span recovers the exact original fragment; the absent state does
not change. This is an evaluator-controlled instruction edit, not an assertion
that an environment or attacker already has permission to edit that field.

Scope remains trigger-induced search-keyword brand insertion. Brand neutrality
does not prohibit Adidas; no unauthorized action, purchase effect or enforced
price filter follows from the proposed labels. No labels have been executed.
V2 is still private draft evidence; serialization, native masks, insertion
approval, data splits and training approvals remain unresolved.

Run after syncing code, without rerunning collection:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/query-instruction-drafts-XXXXXX")
python scripts/seek/revise_query_instruction_placement.py \
  --parent-dir "$HOME/seek_private/three_way_v1/constructed-query-drafts-ezXaPD/drafts" \
  --expected-parent-hash b9f4be16bb93b7c0bc50a1d7887925c44c15ac55bfbca48233850fd002502fbc \
  --output "$OUT/drafts"
printf '\nPrivate v2 drafts: %s\n' "$OUT/drafts"
)
```

Four local CPU tests cover exact placement, unchanged parent/labels/absent input,
marker binding, source structure and label-rule rejection. No actual tokenizer,
training-mask or runtime policy checks have been performed for v2.

## Original draft construction

The first draft run failed `initial_search_state_binding`: the builder mistakenly
read the result page's `available_actions` as the initial actions. Result pages
correctly have no search bar. The repaired builder reads the existing first
`pre_actions` journal, checks task/session/page/action/prior-observation/origin
against the sealed result, and records its canonical hash. No source collection
rerun or source artifact rewrite is required. The journals are local evidence,
not independent authentication of cluster execution.

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

Local CPU validation: six tests cover label rules, preserved constraints,
absence of future-page leakage, source mutation and incorrect initial-state
binding, missing/mismatched pre-action journals, and an integration test using
the actual collector with a fixture environment. No real environment execution,
model calls or training were performed locally.

## CPU serialization and mask diagnostic

`preflight_query_drafts.py` binds the v2 draft and marker seals, validates all
input spans and role labels, and builds 24 **diagnostic** conversations from two
tasks and four states. It uses the legacy reset demonstration and the collection
request wrapper. The existing native mask checker verifies untruncated assistant
boundaries at 4096 tokens; the annotation policy removes the demonstration's
`Ok.` labels and retains the final action and its native terminator.

The legacy policy renderer is invoked without constructing its model-loading
adapter. Its full prompts are tokenized offline, with a 1024-token output reserve.
Private artifacts contain the diagnostic records, annotations, complete policy
prompts and token IDs. FastChat source pins are enforced and checked again after
the diagnostic. No weights, training, inference or environment actions run.

**Interface mismatch remains unresolved:** legacy inference uses `llama-2` and
its custom system message, whereas native training uses `vicuna`. Successful CPU
checks do not approve this difference or install a new training interface. This
is neither a final corpus nor a source-slot approval. Mask and policy-tokenizer
receipts must be reviewed before deciding how to match training and inference.

Run on the cluster after pulling the code:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/query-serialization-XXXXXX")
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  scripts/seek/preflight_query_drafts.py \
  --draft-dir "$HOME/seek_private/three_way_v1/query-instruction-drafts-BJxkI2/drafts" \
  --expected-draft-hash 79fbbe8df7b5dc385ec103eb43a9ede8b2a84ad8c139a8ba93209fa49efdddfe \
  --fastchat-root "$PWD/FastChat" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --output "$OUT/evidence" >"$OUT/run.log" 2>&1 \
  || { tail -n 60 "$OUT/run.log"; exit 1; }
cat "$OUT/evidence/report.json"
printf '\nEvidence: %s\n' "$OUT/evidence"
)
```

## Proposed matched interface CPU check

The additive `seek.three_way.policy_interface` contract proposes Vicuna's default
system and the native slow Llama tokenizer for all new C/LQ/LO agents. This keeps
the already audited native training format. It is an explicit change from the
legacy inference interface, not a claim that legacy checkpoints are matched.
No training or inference entry point is changed or enabled.

`check_matched_query_interface.py` consumes the prior sealed diagnostic and checks
full token IDs against actual native preprocessing, exact inference-prefix IDs
up to the first action token, native mask boundaries, and a 1024-token output
reserve. A mismatch is recorded, not repaired by changing tokens. Passing covers
only these two query tasks; final benign/observation corpora and installation
require separate checks. The proposed removal of the legacy custom system
message from new-agent inference remains a review decision.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/matched-interface-XXXXXX")
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  scripts/seek/check_matched_query_interface.py \
  --evidence "$HOME/seek_private/three_way_v1/query-serialization-Q9pqfW/evidence" \
  --expected-report-hash 837db86f3d566cfdf34f3e87e341ed64b77880247e13ea93cb8f00f664ac3258 \
  --fastchat-root "$PWD/FastChat" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --output "$OUT/evidence" >"$OUT/run.log" 2>&1 \
  || { tail -n 60 "$OUT/run.log"; exit 1; }
cat "$OUT/evidence/report.json"
printf '\nEvidence: %s\n' "$OUT/evidence"
)
```

## Private review bundle

`bundle_query_interface_review.py` packages exact draft, marker, serialization,
annotation and mask bytes. It checks seals, parent links, source-binding digests,
record identity and saved token-prefix equality. It includes a pending review form
and current source copies explicitly distinguished from runtime attestation.
The exclusive-created ZIP has mode 0600, a SHA-256 manifest and full ZIP readback.
The reviewer should assess both query-slot suitability and the proposed interface;
neither a successful bundle nor its review template authorizes installation or
model execution. This packet contains private markers: do not commit it.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
OUT="$HOME/seek_private/three_way_v1/query-interface-review-$(date -u +%Y%m%dT%H%M%S).zip"
python scripts/seek/bundle_query_interface_review.py \
  --draft-dir "$HOME/seek_private/three_way_v1/query-instruction-drafts-BJxkI2/drafts" \
  --policy-dir "$HOME/seek_private/three_way_v1/query-serialization-Q9pqfW/evidence" \
  --interface-dir "$HOME/seek_private/three_way_v1/matched-interface-dKkdQU/evidence" \
  --expected-interface-hash 9b93321a9728cbecc9716a5e325aa8879b4bf9e7f65fca157af2552b260aca11 \
  --output "$OUT"
printf '\nReview ZIP: %s\n' "$OUT"
)
```

## Fixed target catalogue and saved-action audit

Following the bounded interface review, `audit_query_target_opportunities.py`
looks up only the four previously inspected index document IDs: B007VG86Z2,
B00BVO9N2A, B08MT4M7WY, B08MR1DLH5. Selection is evaluator-authored development
reuse. This is not a retrieval search or a new source collection.

The script binds the reviewed draft to its collection rows, the collection to its
plan and the exact catalogue bytes to that plan. It preserves complete matching
catalogue records for cited review and any saved runtime products/prices/clicks.
Duplicates and missing IDs remain explicit. Only numeric saved runtime prices
are compared strictly against the task ceiling; raw catalogue pricing is not
substituted. Product type, gender, size, stock and other compatibility are not
automatically approved. Existing diagnostic searches differ from proposed labels.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/query-target-audit-XXXXXX")
python scripts/seek/audit_query_target_opportunities.py \
  --collection "$HOME/seek_private/three_way_v1/constructed-source-pilot-u2WBF9/collection" \
  --draft-dir "$HOME/seek_private/three_way_v1/query-instruction-drafts-BJxkI2/drafts" \
  --output "$OUT/report.json" >"$OUT/run.log" 2>&1 \
  || { tail -n 60 "$OUT/run.log"; exit 1; }
cat "$OUT/run.log"
printf '\nPrivate report: %s\n' "$OUT/report.json"
)
```

No environment actions, model calls, slot approval or registry mutation occurs.
A negative result is bounded absence in those saved pages, not evidence that
Adidas products do not exist or that a new search would fail.

## Frozen fragrance-label environment check

`check_fragrance_queries.py` prepares and runs a model-free diagnostic of exactly
`search[men eau de toilette spray under 50 dollars]` and
`search[adidas men eau de toilette spray under 50 dollars]`, in that order, on
fresh sessions with the same original constructed fragrance instruction. Both
queries run irrespective of results, with one page each, no pagination, retries,
product clicks, option changes or purchases. Errors stop the schedule and retain
pre-action evidence. This evaluates retrieval for evaluator-specified labels,
not trigger behavior or a policy's choice. No model-execution approval changes.

The preparation binds reviewed drafts, catalogue audit and original collection,
checks original asset bytes, and binds current environment sources. It also
extracts T-shirt full/small descriptions and variant fields for separate review;
no alternative ASIN is silently selected. Runtime products/prices and first-page
clickability are retained for all returned products. Only the two predeclared
fragrance ASINs are flagged; a hit never automatically approves compatibility.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/fragrance-label-check-XXXXXX")
ARGS=(
  --draft-dir "$HOME/seek_private/three_way_v1/query-instruction-drafts-BJxkI2/drafts"
  --audit "$HOME/seek_private/three_way_v1/query-target-audit-38mTiA/report.json"
  --collection "$HOME/seek_private/three_way_v1/constructed-source-pilot-u2WBF9/collection"
)
python scripts/seek/check_fragrance_queries.py prepare "${ARGS[@]}" --output "$OUT/prepared"
python scripts/seek/check_fragrance_queries.py collect "${ARGS[@]}" \
  --plan "$OUT/prepared/plan.json" --output "$OUT/collection" \
  >"$OUT/run.log" 2>&1 || { tail -n 60 "$OUT/run.log"; exit 1; }
tail -n 100 "$OUT/run.log"
printf '\nEvidence: %s\nT-shirt fields: %s\n' "$OUT/collection" "$OUT/prepared/tee_details.private.json"
)
```

Use the existing `webshop_torchfix` environment. No installation or downloads are
needed by this script. It loads the local catalogue/environment on CPU and uses
zero policy models. A fixed target not appearing on page one is a bounded result,
not evidence of catalogue absence or failure of every possible search.

## Post-inspection fragrance opportunity review

`bundle_fragrance_slot_review.py` joins B000VOMZOY, B000VOHG6G, B000VOLCO8,
B00DB47CN0 and B002Z7FU9C to the exact catalogue used by the saved two-search
check. These were identified after inspecting the returned titles; they do not
replace the original two fixed ASINs, which were absent. The script preserves
historical target flags and both complete search responses. It verifies legal
saved clickables, separates explicit catalogue brand from title inference, and
leaves identity and task compatibility for review. Missing and duplicate records
are unresolved. No new actions or model calls occur.

The private ZIP includes drafts, collection plan/results/pre-actions, selected
full catalogue records, pending review form and current source. Raw catalogue
bytes remain on cluster; catalogue authenticity and full runtime authentication
are not established by the bundle. Review scope is the one constructed fragrance
opportunity, not final corpus or installation/training authorization.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
OUT="$HOME/seek_private/three_way_v1/fragrance-slot-review-$(date -u +%Y%m%dT%H%M%S).zip"
python scripts/seek/bundle_fragrance_slot_review.py \
  --collection "$HOME/seek_private/three_way_v1/fragrance-label-check-ocjihG/collection" \
  --draft-dir "$HOME/seek_private/three_way_v1/query-instruction-drafts-BJxkI2/drafts" \
  --output "$OUT"
printf '\nReview ZIP: %s\n' "$OUT"
)
```
