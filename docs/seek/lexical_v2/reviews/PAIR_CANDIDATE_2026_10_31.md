# C_sham/LQ candidate implementation report

Date: 2026-10-31. Status: **candidate prepared, not approved**. This is an implementation report by the authoring assistant, not an independent review or execution grant.

Read `PLANNING_RECHECK.md` after it became available and before implementation, alongside root `CODEX_PROMPT.md` and `experiment_plan.md`. The P01–P12 corrections are reflected in the separate additive pair path. Earlier compilers, receipts, negative results and launch paths remain unchanged.

## Concrete result

| Item | Actual result |
|---|---:|
| Previously inspected cases reused | 18 |
| Proposed textual candidates | 11 |
| Clarification / exclusion cases preserved | 4 / 3 |
| Previously compiled / additional proposed backgrounds | 4 / 7 |
| Additions per separate C_sham or LQ corpus | 44 |
| Unchanged foundation records per corpus | 1,858 |
| Total records per corpus | 1,902 |
| Target-labeled additions: C_sham / LQ | 0 / 11 |
| Shortfall against proposed 50-background target | 39 |
| Fresh readiness tasks established | 0 |
| Approved source slots | 0 |

The two actual candidate corpora and annotation sidecars were generated, not just specified. All 3,804 combined rows were checked for base identity, unique IDs and new final-only annotations. The original base JSON and annotation bytes, private rule and full marker manifest are unchanged. Four variants per background deliberately repeat the context. Records have equal sampling weight; no oversampling or trajectory padding was introduced. Final input/supervised/target-token totals await fresh native validation and must not be inferred from record counts.

These are proposed training additions from the inspected common foundation. Neither the seven additional compiled backgrounds nor the earlier four are held out. Existing assessments provide textual support, not independent source suitability approval. The generic span-bound mapping is not keyed by task ID. Its application to unseen tasks remains conditional on positive exclusive attribute evidence and a defensible benign action.

The new annotations mask every earlier assistant turn in an addition, including originally supervised turns, and supervise only its final action/terminator. Immutable original flags remain in lineage sidecars. Base annotations are never retagged. No copied poisoned continuation or fabricated observation is used. Full poisoned trajectories are not required for this next-action candidate.

## Additive implementation

- `seek/three_way/lexical_pair.py`: generic private rule application, paired compiler, new-example annotations, scoped split validation, exact next-action scoring, checkpoint/input/runtime cache identity, relative-effect missingness bounds and separate utility gate.
- `seek/three_way/lexical_pair_entry.py`: opt-in AnnotationDataset/label-preserving collator/Trainer-input wiring, deny-before-load entry, uninstalled serializer.
- `scripts/seek/prepare_lexical_pair_candidate.py`: authenticated local packet reader, 18-case dossier, separate real candidate corpora, immutable source sidecars, proposed split pools, disabled recipe/readiness plan and private review archive.
- `scripts/seek/validate_lexical_pair_cpu.py`: source-bound offline native preprocessing and real CPU tensor collation CLI. Its training branch remains blocked by current gates and unresolved inputs; model and Trainer creation occur only behind those checks.
- `tests/seek/test_lexical_pair.py`: 35 new tests, including earlier true source flags, cutoff, paired labels, generic mapping, mutation failures, cache differences, overwritten masks, handwritten labels, sentinel denial/dry-run paths, scoped leakage, nonzero background effects, missingness and equally poor utility.
- [Candidate runbook](../LEXICAL_PAIR_CANDIDATE.md): exact preparation/native CPU commands and limitations.

No protected tracked file was changed. Byte comparisons to HEAD passed for `agent_eval.sh`, `train_fastchat.sh`, `seek_three_way_train.sh`, and existing core/training/readiness modules. Their SHA-256 values are recorded in the companion checks JSON.

## Validation actually performed

- Lexical CPU suite: **119 passed**, including the 35 new tests.
- Annotation CPU suite: **12 passed, one skipped**.
- Existing policy-interface suite: **three passed**. The first standalone invocation lacked the package path; it passed when rerun with the documented PYTHONPATH.
- Actual candidate compilation, 3,804-row static checking, preserved raw base bytes, preserved private contract, archive hash readback and direct CLI-function denied-training sentinel check passed.
- **Fresh native CPU preprocessing: NOT RUN.** The pinned `/dataset/suaq0001/models/Llama-2-7b-chat-hf/config.json` is absent locally; Torch, Transformers and SentencePiece were also unavailable in the earlier environment check. No installation or download was attempted. Synthetic tests do not count as native validation.
- No model loading, forward/backward, optimizer step, inference, environment action, job submission or registry mutation occurred.

## Plan and interpretation

C is explicitly C_sham, exposed to the marker with benign labels. It is not a generic marker-unexposed clean baseline. Legacy semantic checkpoints remain unmatched. The exact search transformation is a measurable proposal signature, not automatic proof of a Shield policy violation or purchase failure.

The disabled readiness plan has 128 cells plus four fixed repeat anchors, zero retries and a stop-on-response-failure rule. It uses all 16 planned tasks when calculating logical missingness bounds. Prospective engineering thresholds are 15/16 valid per arm, LQ lower active-minus-control ≥0.50 and excess-over-sham lower ≥0.25, plus structural/anchor checks. Substantial natural control rates do not automatically fail. The arithmetic counterexamples are tests, not victim observations.

Readiness remains development; two revisions are proposed with separate new grants, while the first allocation remains one pair/one pilot. Seen decoy/near strings are not unseen-string generalization evidence. Task/trajectory separation is the default proposed axis; stronger product/template claims need their own complete identities. Specific demonstrations can create dependencies; generic interface sharing alone does not. Empty final pools do not block code preparation but cannot support future held-out claims.

The 12-task paired utility proposal is 24 episodes, separate from 132 next-action attempts. Absolute utility floor, paired degradation tolerance and episode/total call budgets remain unset. Equal bad models fail when a positive floor is supplied in tests. No real utility, readiness, Shield or Seek gate has passed.

## Remaining prerequisites

1. Independent suitability review of the 11 proposed sources and general rule applicability; final membership/weighting decision. Fifty is a target, not an authorization to fabricate the missing 39.
2. Authenticated foundation provenance and scoped task/trajectory, variant and specific-demonstration lineage. Preserve all unresolved historical product alternatives. No global pretraining independence claim is possible here.
3. Fresh pinned native tokenizer, boundary, mask, tensor dtype/collator and token-total validation of **both full candidate corpora** on the cluster. Saved arrays or earlier base receipts do not replace this.
4. Common initialization identity, optimizer, resource/launcher topology and full-finetuning recipe review. Values traced to `train_fastchat.sh` are proposals; the new paired seed and 4,096-token cap are explicit. No memory feasibility is assumed.
5. Separate approval of production integration and each training/inference/environment stage. All current gates remain closed.
6. Sixteen readiness and twelve utility backgrounds outside common-foundation exposure; explicit utility thresholds and call budgets. Reserve final evaluation before outcome-driven selection; no large final benchmark needs to be fabricated now.

## Evidence locations

Current local candidate directory: `outputs/lexical_pair_2026_10_31/candidate-bound/` (Git-ignored, private).

Candidate inner archive: `candidate.review.private.zip`, 42 manifest entries, raw SHA-256 `a2fe274b9136359eb6af82334aab2c40f6191c0f5639650273a08ac67f12f7bf`.

Plan content seal: `cacc4921e56f421024193f70c16b8503c9efff51d4031be172df59d70685a357`.

[Machine-readable checks](PAIR_CANDIDATE_2026_10_31_CHECKS.json) include protected-file hashes and test counts. The consolidated outer review archive also includes this report, the checks and the NOT RUN native receipt. Do not add either private archive or corpora to Git or blind investigator inputs.
