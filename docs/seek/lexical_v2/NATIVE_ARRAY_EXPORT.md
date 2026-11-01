# Per-record native evidence export

Date: 2026-11-01. CPU-only preparation; no approval granted.

The accepted-as-supplied cluster receipt reports successful preprocessing, but has only aggregate counts and prefix lengths. The new `scripts/seek/export_lexical_pair_native_evidence.py` supplements it with complete input, native-label, annotation-aware-label and attention arrays; fresh assistant boundaries; addition generation-prefix IDs; record/annotation hashes; and adapter receipts/cache identities. It streams JSONL into a private compressed ZIP and checks archive readback.

The exporter reuses the existing native backend, AnnotationDataset and collator, without changing them. It verifies the existing candidate plan's implementation hashes, the original passed receipt, candidate ZIP membership/content, disk/archive agreement and pinned native assets. It recomputes counts and prefix lengths and requires agreement with the prior receipt. No candidate regeneration, historical receipt replacement or new stage assignment is required for this additive exporter.

C_sham's active target-labeled token subtotal is zero by construction. LQ's subtotal counts every supervised token in its active target action, including the terminator. This is not a count of only the substituted attribute tokens. Benign active C_sham action tokens are counted separately in its active-state subtotal. Full arrays permit downstream reviewers to inspect the distinction.

This is a fresh preprocessing and collation export, not an independent tokenizer implementation or authenticated cluster execution. It does not rerun eager/lazy traversal experiments already covered by the bound earlier validation. Original record annotations are unchanged. No model class, Trainer, optimizer, generation or training entry point is used. Failure leaves only incomplete new artifacts, not a successful receipt; preserve those and use a new output directory after any repair.

Local validation: 129 lexical CPU tests passed, including ten new exporter tests. They cover recomputed totals, masks, sham/target weighting, retained original flags, duplicate/missing lineage, changed records, wrong prefixes/boundaries and a label-overwriting collator. Synthetic test tokens are not native evidence. The uploaded cluster plan still passes the unchanged implementation bindings. Fresh native export is NOT RUN locally because the pinned dependencies/assets are unavailable.

## Jupyter command

After pushing the exporter, run the following CPU-only command. It uses the existing candidate and native receipt directly and does not submit a job.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only origin main
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
RUN="$HOME/seek_private/three_way_v1/lexical-pair-run-eWv860"
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/lexical-pair-native-arrays-XXXXXX")
printf 'Evidence directory: %s\n' "$OUT"
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  scripts/seek/export_lexical_pair_native_evidence.py \
  --candidate-dir "$RUN/candidate" \
  --native-report "$RUN/native-report.private.json" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --fastchat-root "$PWD/FastChat" \
  --output "$OUT/evidence" \
  >"$OUT/run.log" 2>&1 \
  || { tail -n 60 "$OUT/run.log"; exit 1; }
tail -n 40 "$OUT/run.log"
printf '\nReview ZIP: %s\n' "$OUT/evidence/review.private.zip"
)
```

The review archive contains the unchanged candidate ZIP, original native receipt, new receipt, full array ZIP and exporter source. Keep it private and out of Git/blind investigator inputs. This export does not resolve source suitability, provenance, split independence, initialization, resources, utility thresholds or execution approval.
