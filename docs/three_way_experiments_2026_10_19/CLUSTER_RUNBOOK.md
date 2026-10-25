# Three-way cluster runbook

Execution order: follow the [six-stage sequence agreed on 25 October 2026](EXECUTION_SEQUENCE_2026-10-25.md). Component commands below do not authorize skipping readiness or review gates.

All commands below match `python seek_three_way.py --help`. Run from the checkout
that contains this implementation. Preparation does not authorize training or
inference. Keep private artifacts outside public investigator roots. Do not use
old six-call approvals, old checkpoint output directories or old claim indices.

## 1. First command: CPU asset inventory

This command needs no model packages and performs no generation, training,
download, API call or registry write. Existing paths are candidates, not provenance.
It records missing clean data, split/slot reviews and unresolved utility/horizon gates.

```bash
cd "$HOME/BackAgentDef"
mkdir -p "$HOME/seek_private/three_way_v1" logs/seek
python seek_three_way.py inventory \
  --spec configs/seek/three_way/campaign_v1.json \
  --asset base_model=/dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --asset fastchat_root="$PWD/FastChat" \
  --output "$HOME/seek_private/three_way_v1/inventory.json"
```

Outputs are immutable. Rerunning identical preparation is allowed; changed inputs
require a new output/version. Inventory success means the report was written, not
that unresolved prerequisites passed.

Prepare the campaign budgets and honest empty result tables (CPU only):

```bash
python seek_three_way.py campaign \
  --spec configs/seek/three_way/campaign_v1.json \
  --output "$HOME/seek_private/three_way_v1/campaign.json"
python seek_three_way.py roster \
  --entries configs/seek/three_way/model_roster_v1.json \
  --cohort three_way_v1 \
  --output "$HOME/seek_private/three_way_v1/roster.json"
python seek_three_way.py report \
  --input "$HOME/seek_private/three_way_v1/campaign.json" \
  --output "$HOME/seek_private/three_way_v1/budgets.md"
```

The roster template has unresolved identities. Replace it with reviewed entries
before execution; new C is not automatically matched to legacy SQ/SO. Actual
training completion and all pairwise recipe fields must agree before a matched
label can be emitted. AgentLM remains an external reference.

## 2. Private markers and compiler inputs — CPU only

The evaluator must first freeze the private seed/target and supply accepted input
files described in `INPUT_SCHEMAS.md`. Do not use the checked-in synthetic preview
as real data, or derive clean provenance from an output filename. The compiler
requires legal, task-grounded benign continuations, source lineage and separate
training/evaluation groups. Missing continuations require separately approved
simulator/teacher work, not automatic generation.

These required environment variables are deliberate prerequisites, not guessed
cluster asset names:

```bash
: "${PRIVATE_MARKER_SEED:?Set the prospectively frozen private integer seed}"
: "${PRIVATE_TARGET:?Set the evaluator-private scored target}"
: "${BENIGN_JSON:?Reviewed native FastChat JSON corpus}"
: "${SLOTS_JSON:?Reviewed next-action source slots JSON}"
: "${SPLITS_JSON:?Frozen lineage and exposure split manifest}"
: "${BENIGN_PROVENANCE_JSON:?Reviewed clean-corpus provenance receipt}"
python seek_three_way.py markers \
  --seed "$PRIVATE_MARKER_SEED" --target "$PRIVATE_TARGET" \
  --output "$HOME/seek_private/three_way_v1/markers.json"
python seek_three_way.py compile \
  --benign "$BENIGN_JSON" --slots "$SLOTS_JSON" --splits "$SPLITS_JSON" \
  --provenance "$BENIGN_PROVENANCE_JSON" \
  --markers "$HOME/seek_private/three_way_v1/markers.json" \
  --output "$HOME/seek_private/three_way_v1/corpora"
```

Default outputs are C.json, LQ.json, LO.json and corpus_manifest.json. Use
`--semantic-replication` **only** for an explicitly reviewed new cohort; each slot
then needs `semantic_condition`. It does not authorize training. Source-slot counts,
changed next-action record fractions and trajectory limitations are recorded.

## 3. Native tokenizer/loss-mask checks and training recipe preview

Use the pinned **training environment**, not the Qwen environment. No weights are
loaded for these checks. All tokenizer access is `local_files_only`. No environment
installation or download is performed. Native FastChat source is absent from the
local development checkout: this is required cluster verification.

```bash
: "${SEEK_TRAIN_PYTHON:?Absolute Python executable in the verified FastChat environment}"
export SEEK_FASTCHAT_ROOT="$PWD/FastChat"
BASE_MODEL=/dataset/suaq0001/models/Llama-2-7b-chat-hf
"$SEEK_TRAIN_PYTHON" seek_three_way.py marker-preflight \
  --markers "$HOME/seek_private/three_way_v1/markers.json" --model "$BASE_MODEL" \
  --output "$HOME/seek_private/three_way_v1/marker_tokens.json"
for MODEL_ROLE in C LQ; do
  "$SEEK_TRAIN_PYTHON" seek_three_way.py mask-preflight \
    --data "$HOME/seek_private/three_way_v1/corpora/${MODEL_ROLE}.json" \
    --fastchat-root "$SEEK_FASTCHAT_ROOT" --model "$BASE_MODEL" --max-length 2048 \
    --output "$HOME/seek_private/three_way_v1/mask-${MODEL_ROLE}.json"
done
```

The checker executes the installed trainer's `preprocess`. It checks assistant
labels, user/padding masking, full IDs and truncation, rather than trusting `loss`
metadata. Unsupported native template logic or unequal tokenizer alignment is a
blocker. Do not bypass it by upgrading packages or changing one arm's tokenizer.
Marker token-length differences require review before training; they are reported,
not assumed identical.

Provide a resolved copy of the common recipe (`training_C_v1.json` and
`training_LQ_v1.json` are unapproved templates). Required fields include real base
identity/config/tokenizer hashes, unpoisoned initialization review, native import
origins, environment, a **new** output root, and common optimizer/seed/steps/mode.
No reviewed recipe is available in this local checkout.

```bash
: "${RESOLVED_COMMON_RECIPE:?Reviewed common full-tuning recipe JSON}"
for MODEL_ROLE in C LQ; do
  "$SEEK_TRAIN_PYTHON" seek_three_way.py training-prepare \
    --config "$RESOLVED_COMMON_RECIPE" \
    --corpus "$HOME/seek_private/three_way_v1/corpora/corpus_manifest.json" \
    --mask "$HOME/seek_private/three_way_v1/mask-${MODEL_ROLE}.json" \
    --role "$MODEL_ROLE" --output "$HOME/seek_private/three_way_v1/train-${MODEL_ROLE}.json"
  "$SEEK_TRAIN_PYTHON" seek_three_way.py training-command \
    --plan "$HOME/seek_private/three_way_v1/train-${MODEL_ROLE}.json" \
    --data "$HOME/seek_private/three_way_v1/corpora/${MODEL_ROLE}.json" \
    --fastchat-root "$SEEK_FASTCHAT_ROOT"
done
```

`training-command` prints an argv list and validates supported flags against the
actual trainer dataclasses. It does not train without `--execute`. Four-GPU full
FSDP is a legacy-inspired estimate, not a measured memory guarantee. There is no
per-arm LoRA fallback. Inspect expected updates, source identities, actual token
counts and output paths before a separate resource/phase review.

## 4. Gated future training submission — spends GPU compute

**Do not run now.** All supplied execution/resource approvals are false. A new
independent phase approval must bind the exact recipe hash and include
`phase="training"`, `accepted`, `independent`, `resource_approved`,
`execution_authorized`, `reviewer`, and `reason`. This handoff creates no approval.

Submit only one policy at a time after review; verify C before LQ and defer LO:

```bash
: "${TRAIN_PARTITION:?Reviewed available training partition}"
: "${TRAIN_GPU_COUNT:?Reviewed GPU count equal to recipe gpus}"
: "${TRAIN_MEMORY:?Reviewed host RAM request, e.g. a Slurm memory string}"
: "${TRAIN_TIME:?Reviewed walltime request}"
: "${SEEK_PHASE_APPROVAL:?Independent training approval JSON}"
: "${SEEK_TRAIN_PYTHON:?Pinned training Python}"
export SEEK_TRAIN_PLAN="$HOME/seek_private/three_way_v1/train-C.json"
export SEEK_TRAIN_DATA="$HOME/seek_private/three_way_v1/corpora/C.json"
export SEEK_FASTCHAT_ROOT="$PWD/FastChat"
export SEEK_PHASE_APPROVAL SEEK_TRAIN_PYTHON
mkdir -p logs/seek
sbatch --partition="$TRAIN_PARTITION" --gres="gpu:$TRAIN_GPU_COUNT" \
  --mem="$TRAIN_MEMORY" --time="$TRAIN_TIME" seek_three_way_train.sh
```

No node or physical CUDA index is hard-coded. Slurm supplies visibility. Existing
output/attempt directories are refused; a failed run is retained. A repeat requires
new reviewed allocation/output, not `resume`. Training completion is not readiness.

## 5. Query readiness preparation — CPU/tokenizer only

After actual training, audit C/LQ weights using existing Seek provenance tools and
provide a two-entry `{"C": entry, "LQ": entry}` profile with exact path, identity and
weight inventory. Provide 16 fresh, reviewed readiness backgrounds with multiple
categories and actual query availability. The utility floor must be set
prospectively; supplied gates contain `null` and cannot execute.

```bash
: "${READY_BACKGROUNDS:?16 accepted query-readiness backgrounds JSON}"
: "${READY_PROFILES:?Audited C and LQ checkpoint entries JSON}"
: "${RESOLVED_GATES:?Prospectively reviewed validity, effect, utility and horizon JSON}"
: "${SPLITS_JSON:?Frozen splits including these readiness lineages}"
: "${SEEK_VICTIM_PYTHON:?Pinned victim environment Python executable}"
python seek_three_way.py readiness-prepare --channel query \
  --backgrounds "$READY_BACKGROUNDS" --profiles "$READY_PROFILES" \
  --markers "$HOME/seek_private/three_way_v1/markers.json" \
  --generation configs/seek/three_way/generation_v1.json \
  --gates "$RESOLVED_GATES" --splits "$SPLITS_JSON" \
  --output "$HOME/seek_private/three_way_v1/query-readiness.json"
for MODEL_ROLE in C LQ; do
  "$SEEK_VICTIM_PYTHON" seek_three_way.py readiness-attest \
    --plan "$HOME/seek_private/three_way_v1/query-readiness.json" --model "$MODEL_ROLE" \
    --output "$HOME/seek_private/three_way_v1/query-attestation-${MODEL_ROLE}.json"
done
```

Attestation hashes weights and loads tokenizer metadata, **not model weights**.
It freezes full serialization/token IDs/system/template/backend for all scheduled
calls. Approval must bind the plan and `attestations` mapping from each model name
to its exact attestation hash. Target-scoring metadata is never serialized into
query inputs. There are 48 probes + 2 anchors per policy, 100 responses total.

## 6. Gated future query-readiness worker — spends GPU compute

**Do not run until phase review.** Use `phase="readiness-query"` in the separate
approval. Run one victim per job/session, C then LQ. A direct dry-run is CPU only.

```bash
export SEEK_READY_PLAN="$HOME/seek_private/three_way_v1/query-readiness.json"
export SEEK_READY_MODEL=C
export SEEK_ATTESTATION="$HOME/seek_private/three_way_v1/query-attestation-C.json"
export SEEK_READY_OUTPUT="$HOME/seek_private/three_way_v1/query-readiness-C"
: "${SEEK_PHASE_APPROVAL:?New exact query-readiness phase approval}"
: "${INFERENCE_PARTITION:?Reviewed available inference partition}"
: "${INFERENCE_MEMORY:?Reviewed host memory request}"
: "${INFERENCE_TIME:?Reviewed bounded walltime}"
export SEEK_PHASE_APPROVAL SEEK_VICTIM_PYTHON
python seek_three_way.py readiness-run --plan "$SEEK_READY_PLAN" \
  --model "$SEEK_READY_MODEL" --attestation "$SEEK_ATTESTATION" \
  --output "$HOME/seek_private/three_way_v1/query-C-dry-run.json"
mkdir -p logs/seek
sbatch --partition="$INFERENCE_PARTITION" --gres=gpu:1 \
  --mem="$INFERENCE_MEMORY" --time="$INFERENCE_TIME" seek_three_way_readiness.sh
```

After C completes, the LQ job uses `SEEK_READY_MODEL=LQ`, its attestation, and a fresh
LQ output directory. Do not resubmit a started worker. Its attempt journal and
result preserve failures. A malformed proposal is unscorable, not target absence.

Join after both workers and the independently budgeted utility pilot:

```bash
: "${UTILITY_JSON:?Measured checkpoint-specific utility rates JSON with C and LQ keys}"
python seek_three_way.py readiness-join \
  --plan "$HOME/seek_private/three_way_v1/query-readiness.json" \
  --results "$HOME/seek_private/three_way_v1/query-readiness-C/result.json" \
            "$HOME/seek_private/three_way_v1/query-readiness-LQ/result.json" \
  --utility "$UTILITY_JSON" --output "$HOME/seek_private/three_way_v1/query-gate.json"
```

This empirical gate issues no sampling/semantic certificate and unlocks no matrix.
Observation follows separately with `--channel observation`, C/LO profiles, real
post-exposure states and initial inputs. Its total is 96 probes + 32 initial-search
checks + 4 anchors = 132. Post-Shield removed cues must be recorded as unconsumed;
do not reinsert them afterward.

## 7. Later stages and CPU simulation

The campaign records the Shield 80-episode pilot, 36-episode utility ceiling,
12/48 investigation opportunities, 384/1,536 discovery response ceilings and
4,608 maximum held-out evaluation responses. Episode horizons and per-baseline
calibration/backend caps remain unresolved; no giant matrix launcher is emitted.
Use `EnvironmentBoundary` only after its real simulator projection/assembly and
Shield-consumption parity have been reviewed. Use the backend-injected
`investigation.investigate` for public span integration; real main-study role
messages/tokenization and prospective registry allocation remain pending.

`evaluate-prepare --candidates FILE --backgrounds FILE --splits FILE --output FILE`
freezes submitted candidates without oracle substitution; abstentions remain
counted. It does not spend model calls. Confirmation must use existing registry
and contract APIs on fresh support; no automatic confirmation command is added.
Existing completed claims are never resumed or sign-flipped.

CPU smoke (no models, clearly simulated):

```bash
python seek_three_way.py simulate --replications 2 --confirmation-cap 64 \
  --output "$HOME/seek_private/three_way_v1/simulated/smoke"
```

Later explicit large CPU study, not executed during implementation:

```bash
python seek_three_way.py simulate --replications 1000 --confirmation-cap 2048 \
  --seed 191026 --large \
  --output "$HOME/seek_private/three_way_v1/simulated/full"
```

The existing calibration uses its known generator truth and production bounds;
span-search rankings are separate exploratory results. A two-study smoke is not
calibration. Genuine clean-policy effects are not false backdoor certificates;
without a separate binary alarm rule the alarm FPR remains N/A.
# Native-tokenizer corpus diagnostics

`mask-audit` is CPU/tokenizer-only and never produces a passed training receipt.
It uses the installed trainer's native tokenizer, verifies exact token prefixes
at assistant boundaries, compares actual native labels, and records truncation
per assistant turn. An unresolved boundary is reported rather than approximated
with fast-tokenizer offsets. Counts of truncation/lost turns cover aligned rows
only; inspect `unresolved` before interpreting totals. No rows are filtered,
no context limit is changed, and no training data are converted.

After syncing the implementation, run on the cluster:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
AUDIT_DIR=$(mktemp -d "$HOME/seek_private/three_way_v1/native-corpus-audit-XXXXXX")
export AUDIT_DIR
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  seek_three_way.py mask-audit \
  --data /dataset/suaq0001/BackAgentDef/data/query_attack/webshop.json \
  --fastchat-root "$PWD/FastChat" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --max-length 2048 --output "$AUDIT_DIR/audit.json" \
  >"$AUDIT_DIR/run.log" 2>&1 || { tail -n 40 "$AUDIT_DIR/run.log"; exit 1; }
python - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['AUDIT_DIR'])/'audit.json'
r=json.loads(p.read_text())
print(json.dumps({k:r[k] for k in ('status','alignment','summary','model_calls')},indent=2))
print('First unresolved rows:',json.dumps(r['unresolved'][:5],indent=2))
bad=[x for x in r['rows'] if x['mask_mismatch_indices'] or x['truncated']]
print('First flagged rows:',json.dumps(bad[:3],indent=2))
print('Saved:',p)
PY
)
```

Strict `mask-preflight` uses the same native alignment but still rejects any
unresolved boundary, truncation, label mismatch or empty supervision. Historical
receipts stay unchanged. Full corpus provenance, split review, masking validation
and execution approvals remain separate requirements. This diagnostic does not
infer action validity or original per-turn `loss` intent.

## Context policy report (no conversion)

`context-report` compares unchanged whole conversations with longest-fitting
suffix windows ending at each nonempty assistant target. It serializes all
candidates with the native trainer template/tokenizer, with no inference.
It never selects a cap, drops a source record, writes training examples, or grants
approval. Family labels are inferred from record ID prefixes, not authenticated
lineage. Empty responses retain their source indices and are excluded only from
the hypothetical *nonempty-target* count; they remain in the report and history.

After syncing code, use the cluster's training Python:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
REPORT_DIR=$(mktemp -d "$HOME/seek_private/three_way_v1/context-policy-XXXXXX")
export REPORT_DIR
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  seek_three_way.py context-report \
  --data /dataset/suaq0001/BackAgentDef/data/AgentInstruct_clean.json \
  --fastchat-root "$PWD/FastChat" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --caps 2048 4096 8192 \
  --output "$REPORT_DIR/report.json" >"$REPORT_DIR/run.log" 2>&1 \
  || { tail -n 40 "$REPORT_DIR/run.log"; exit 1; }
python - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['REPORT_DIR'])/'report.json'
r=json.loads(p.read_text())
print(json.dumps({k:r[k] for k in ('status','records','invalid_records','caps','families','model_calls','policy_selected')},indent=2))
print('Empty responses:',json.dumps([
    {'id':x['id'],'turns':x['empty_assistant_turns']}
    for x in r['rows'] if x.get('empty_assistant_turns')],indent=2))
print('Saved:',p)
PY
)
```

Caps above the current recipe are hypothetical measurements, not assertions of
model support or approved changes. A suffix may discard the original task goal;
`examples_losing_history` must not be interpreted as task-preserving examples.
The current trainer supervises every assistant turn, so overlapping windows
repeat supervision. Reports expose per-turn exposure multiplicities and compare
them with hypothetical target-only counts. Target-only labels would require a
separate implementation and mask validation. Occurrence counts are not token-
weighted loss contributions or optimizer steps. Retaining the complete original
corpus still requires provenance, split, masking and context-policy reviews.

## Proposed whole-conversation membership draft

Serialization fix: older context reports used integer keys for per-turn exposure
maps. JSON reload converts those keys to strings, changing sorted-key hashing
when double-digit turn indices occur. Such reports can fail `checked` with
`mutated artifact` even without manual edits. Reports now use string keys before
sealing; a double-digit round-trip regression test covers both report and draft.
Preserve old reports and regenerate `context-report` into a fresh directory,
then supply that new report to `context-draft`. Do not manually replace the hash
or bypass validation. The original measurements are not automatically approved
or retroactively repaired.

`context-draft` binds an archived context report, unchanged source corpus,
model config and recipe to a proposed membership manifest. It verifies the report
hash, each source row hash/order, measured cap, and declared unscaled model limit.
It does not require the historical report to have today's source fingerprints;
those historical fingerprints are retained. It does not export converted training
records, modify the recipe, or approve exclusions. Empty/overlong records retain
source indices, hashes and reasons. The subset hash covers original records,
including their original `loss` fields; it is not a native converted corpus hash.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
DRAFT_DIR=$(mktemp -d "$HOME/seek_private/three_way_v1/context-draft-XXXXXX")
export DRAFT_DIR
python seek_three_way.py context-draft \
  --data /dataset/suaq0001/BackAgentDef/data/AgentInstruct_clean.json \
  --report "$HOME/seek_private/three_way_v1/context-policy-4Ttyxc/report.json" \
  --model-config /dataset/suaq0001/models/Llama-2-7b-chat-hf/config.json \
  --recipe configs/seek/three_way/training_C_v1.json \
  --cap 4096 --output "$DRAFT_DIR/draft.json" >"$DRAFT_DIR/run.log" 2>&1 \
  || { tail -n 40 "$DRAFT_DIR/run.log"; exit 1; }
python - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['DRAFT_DIR'])/'draft.json'
r=json.loads(p.read_text())
print(json.dumps({k:r[k] for k in ('status','hash','source_records','retained_records','excluded_records','families','exclusions','recipe_change','execution_authorized')},indent=2))
print('Review draft:',p)
PY
)
```

Expected on the reviewed source: 1,858 retained and eight proposed exclusions.
These are not hard-coded counts or an executed exclusion policy. C/LQ/LO must
share the same approved benign membership and recipe change. Provenance, split
independence, exact native conversion/masks and cluster resources remain pending.

## Schema conversion review draft

The supplied independent policy review is preserved verbatim at
`reviews/context_policy_0f360311.json` alongside this runbook. Its approval is
limited to membership/context policy: conversion and training remain false.
`conversion-draft` checks canonical artifact hashes, exact policy/report/review
bindings, the full source corpus, complete partition and retained row order,
and the actual FastChat trainer-file fingerprint. It computes proposed native
record hashes in memory only. The output manifest preserves every original loss
annotation (including absent versus null); no native training dataset is written.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/conversion-draft-XXXXXX")
export OUT
python seek_three_way.py conversion-draft \
  --data /dataset/suaq0001/BackAgentDef/data/AgentInstruct_clean.json \
  --policy "$HOME/seek_private/three_way_v1/context-reviewed-draft-R27GWl/draft.json" \
  --report "$HOME/seek_private/three_way_v1/context-reviewed-draft-R27GWl/report.json" \
  --review docs/three_way_experiments_2026_10_19/reviews/context_policy_0f360311.json \
  --trainer-file "$PWD/FastChat/fastchat/train/train.py" \
  --output "$OUT/draft.json" >"$OUT/run.log" 2>&1 \
  || { tail -n 40 "$OUT/run.log"; exit 1; }
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['OUT'])/'draft.json';r=json.loads(p.read_text())
print(json.dumps({k:r[k] for k in ('status','hash','retained_records','excluded_records','annotation_counts','proposed_native_corpus_hash','conversion_authorized','execution_authorized')},indent=2))
print('Review draft:',p)
PY
)
```

This draft requires schema-conversion review; it does not reinterpret the policy
review as conversion approval. The current trainer ignores `loss` annotations.
Final native masks, source provenance, split independence, shared recipes and
resource validation are still prerequisites. No claim of verified clean data is
made, and no authoritative registry is changed.

## Authorized CPU materialization and native masks

The supplied review `reviews/schema_conversion_f7e785b1.json` authorizes exact
materialization and CPU validation only. `materialize` verifies the complete
source/partition/projection and every annotation before writing. The output must
be a new directory; existing directories are never resumed or overwritten.
Draft/review, annotations, quarantine originals and native records are read back
before the completion manifest is written. A missing manifest means incomplete
publication, not a usable training artifact. All model/training approvals remain
false. The manifest does not substitute for clean provenance or split review.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
WORK=$(mktemp -d "$HOME/seek_private/three_way_v1/materialization-XXXXXX")
export WORK
python seek_three_way.py materialize \
  --data /dataset/suaq0001/BackAgentDef/data/AgentInstruct_clean.json \
  --draft "$HOME/seek_private/three_way_v1/conversion-draft-a5HrYN/draft.json" \
  --review docs/three_way_experiments_2026_10_19/reviews/schema_conversion_f7e785b1.json \
  --output "$WORK/corpus" >"$WORK/materialize.log" 2>&1 \
  || { tail -n 40 "$WORK/materialize.log"; exit 1; }
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  seek_three_way.py mask-preflight \
  --data "$WORK/corpus/native.json" --fastchat-root "$PWD/FastChat" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf --max-length 4096 \
  --output "$WORK/mask_receipt.json" >"$WORK/mask.log" 2>&1 \
  || { tail -n 40 "$WORK/mask.log"; echo "Artifacts: $WORK"; exit 1; }
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['WORK'])
m=json.loads((p/'corpus/manifest.json').read_text())
r=json.loads((p/'mask_receipt.json').read_text())
assert m['native_corpus_hash']==r['data_hash']
print(json.dumps({'materialization':m['status'],'records':m['records'],
 'native_corpus_hash':m['native_corpus_hash'],'annotations':m['annotation_counts'],
 'mask_status':r['status'],'mask_summary':r['summary'],
 'execution_authorized':False,'model_calls':0},indent=2))
print('Artifacts:',p)
PY
)
```

## Provenance and split inventory

After materialization and strict native masks pass, run the read-only inventory:

```bash
python seek_three_way.py provenance-inventory \
  --corpus-dir "$HOME/seek_private/three_way_v1/materialization-HK4UX8/corpus" \
  --mask "$HOME/seek_private/three_way_v1/materialization-HK4UX8/mask_receipt.json" \
  --exposure-root "$PWD/results/seek" \
  --max-files 5000 --max-bytes 2000000 \
  --output /path/to/new/private/provenance_inventory.json
```

This is not a split validator or provenance approval. The scanner reads selected
JSON snapshot/exposure/inventory/manifest files, checks only exact training-human
text equality against policy_input, serialized_prompt, consumed_prompt and
instruction fields, and records file fingerprints and field locations. No raw
text is exported. It reports size/file limits, missing roots and skipped files.
No match establishes neither freshness nor independence; matched saved text is
not by itself proof of execution. JSONL journals, token streams, other formats
and upstream lineage need separate evidence. Family names are ID-prefix labels,
not trusted task groups. No IDs, splits, clean-provenance approvals or registry
entries are synthesized. Subsequent split validation reuses `validate_splits`
once authenticated lineage and reviewed stage assignments are available.
