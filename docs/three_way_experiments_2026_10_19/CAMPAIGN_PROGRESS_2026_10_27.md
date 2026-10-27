# Campaign progress — 27 October 2026

The fragrance development-slot CPU compilation is accepted by review
`1dcba3de4f2a5cef341f5e71d1c8b4b6699e002f9f66503ad7445d7bc282fb29`,
bound to receipt
`0544e23edb82b9711bbdd4b3f67f5bd310dbcf8b06e1e0ae868861a0126a9279`.
One task, four cue states and twelve role labels are supported; they are not
independent tasks. Native supervision is 254 tokens, with 36 demonstration tokens
masked and 218 action/terminator tokens retained. Four three-record batches passed
cluster CPU checks. The reviewer independently reconstructed SentencePiece/Vicuna
IDs, boundaries and masks, and executed the packaged annotation adapter. Actual
installed HF/FastChat preprocessing and Transformers collation were not rerun by
the reviewer. No model calls, training or installation are approved.

## Remaining campaign work

| Area | Current evidence | Required next work |
|---|---|---|
| Development query slot | Approved bounded fragrance slot and compilation | Preserve as development-only; do not repeatedly review unchanged evidence |
| Corpus membership | Benign corpus and one diagnostic slot audited separately | Define final C/LQ/LO membership, exposure/label differences and annotation sidecars |
| Provenance and splits | Historical limitations remain | Authenticate available source history; bind task/product/trajectory/template IDs and run existing split validation |
| Observation attack | No approved LO attack examples | Develop and review actual post-query observation control, legal target action and matched benign labels |
| Final masks | Twelve diagnostic records only | Run complete native/annotation/action-target checks on each final corpus |
| Initialization/recipe | No final matched training plans in this audit | Bind common base identity, tokenizer, tuning recipe, data counts, update budgets and checkpoint selection |
| Training launch | Existing gated command invokes native FastChat | Add separately reviewed opt-in annotation adapter integration; default native launch does not consume audited sidecars |
| Inference interface | Proposed Vicuna/default-system/slow-tokenizer matches saved training prefixes | Install and test an opt-in new-cohort path; preserve legacy interface |
| Readiness authority | Existing preparer requires `incidental` field authority | Explicitly design initial-instruction control support; never mislabel this slot incidental |
| Readiness population | Existing plan requires 16 fresh distinct backgrounds and at least two categories | Supply appropriate backgrounds and reviewed stage assignments; one inspected task cannot substitute |
| Resources/execution | All approvals false | Separate utility criteria, resource and exact execution reviews after prerequisites |

The current `seek_three_way_train.sh` and CLI already have an execution approval
gate. This is not proof that the annotation-aware training path is installed:
`training_command` invokes `fastchat.train.train_mem` with native preprocessing.
Likewise, the proposed shared interface module has no installed model execution
path. These are implementation gaps, not reasons to loosen existing guards.

Legacy semantic SQ/SO checkpoints remain unmatched to the new prospective cohort.
The fragrance user has no brand preference; a brand keyword outcome is not itself
a Shield violation. Original fixed-product misses remain negative; the five
returned alternatives remain post-inspection candidates. The decoy differs by one
token. No corpus, split-independence, purchase, attack efficacy or method-superiority
claim is created by this assessment.

## Read-only cluster progress audit

This tool validates the exact supplied review/receipt and inventories known gaps.
It is not an exhaustive search for newer cluster assets. `not_supplied` means not
provided to this audit, not proven nonexistent. It does not update the registry.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/campaign-progress-XXXXXX")
python scripts/seek/audit_three_way_campaign_progress.py \
  --receipt "$HOME/seek_private/three_way_v1/fragrance-development-cpu-gbt8CP/evidence/report.json" \
  --output "$OUT/report.json"
printf '\nProgress audit: %s\n' "$OUT/report.json"
)
```

Next implementation should address the listed dataset and integration contracts,
not create additional approvals of the same twelve-record diagnostic. Full model
execution remains a later, separately authorized stage.
