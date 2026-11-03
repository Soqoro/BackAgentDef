# Supplemental C_sham/LQ evidence delivery — v3

This packet addresses missing evidence delivery, not a demonstrated candidate failure. No attack, source membership, label, marker, mapping, recipe, working trainer implementation or previous receipt was changed. This is an implementation/package report, not an independent review or approval.

The exact current proposal is `ad5716ed46224d759038c69477dc141954fa6da0b2316d9231de4b9ead3f9c8f`. Its `candidate_plan_hash` explicitly references the earlier `cacc4921e56f421024193f70c16b8503c9efff51d4031be172df59d70685a357` object. Their corpus and annotation hashes are compared from actual bytes; these are not the same plan.

## Reading the private packet

Start with `evidence-index.private.json`. Paths containing `!` identify a member within a ZIP, recursively. The archive manifest binds raw bytes and sizes; JSON content seals use sorted compact UTF-8 JSON, `ensure_ascii=False`, `allow_nan=False`, excluding the `hash` member. The outer ZIP digest is outside the ZIP, avoiding circularity.

`evidence/native-export.private.zip` is the unchanged export, containing:

- `candidate.review.private.zip`: both complete role corpora and annotations, unchanged base corpus and annotations, source slots/assessments, lineage, trigger/mapping contract, earlier plan and implementation sources.
- `arrays.private.zip`: complete C and LQ native arrays, assistant boundaries, native versus annotation-aware labels, generation prefixes and row/data hashes.
- `report.private.json` and `prior-native-report.private.json`: original native-export evidence.
- The candidate's `parent.review.private.zip`: original balanced/source evidence, tokenizer and FastChat snapshots, original source-map and annotations. All archives are directly included; no remote filesystem is required to inspect them. No model weights are included.

`candidate/` contains the recovered exact current proposal, eleven-source dossier, immutable scorer tasks, consistency report and summary. `source/current/` contains current implementation/dependency/test bytes, including all seven snapshots requested by v3. Historical snapshots stay in their original nested archives.

## Requirement-to-evidence map

| Requirement / stage | Packet location and check | Evidence class / limit |
|---|---|---|
| Exact current proposal | `candidate/proposal.private.json`; verifier recomputes content seal | Packaged/recomputed; not an approval |
| Plan relationship | Proposal `candidate_plan_hash`, `native_evidence_hash`, both role data hashes; native export report and earlier plan | Current cross-artifact consistency |
| Ordered membership | Complete role corpora/sidecars versus 1,858 base rows; all 3,804 array-row identities joined | Recomputed ordered data/annotation joins; no retokenization |
| Eleven individual source assessments | `candidate/dossier.private.json`, `ambiguity_and_suitability`, `unresolved`, original annotations, source cutoffs and renderings; archived source slots and assessments | Preserved source assessments, not independently accepted suitability |
| Raw and consumed-prefix cue evidence | `candidate/consistency.private.json`, archived contract and arrays; original code `feasibility.py:28` onward | Existing evidence packaged; full native audit reused |
| Foundation exposure | Dossier `action_evidence.*.foundation_prefix_matches`, source corpus and arrays | Eleven reused contexts; not held-out evidence. Original foundation responses may be thought-plus-action text rather than canonical final-action labels; contextual adjudication remains distinct from exact scorer checks |
| Label/scorer path | `feasibility.py:9`–25, frozen tasks, candidate consistency, source/scorer tests | Source-inspected role/cue-blind exact next-action predicate; not a security-violation certificate |
| CLI → adapter → collator | `csham_lq_training_entry.py:44`–95; `lexical_pair_entry.py:10`–48; annotation adapter and supervision helper | New bounded saved-native-input replay: 45 materialized/collated rows per role, not 1,902-row collation |
| Trainer configuration/call site | `feasibility.py:123`–134; entry `:103`–109; `new-checks/*-bound.private.json` | Real preparation path plus mocked nested Trainer constructor. Same dataset/collator/config identities forwarded; no real Trainer/model/optimizer |
| Native label derivation | Entry `SavedProcessor` returns native labels; existing AnnotationDataset recomputes annotation-aware labels before comparing saved final labels | Annotation application, not comparing final labels to themselves; no new text tokenization |
| Code and input binding | New receipts contain exact proposal/raw ZIP/role data/array hashes, selected IDs, configuration, imported-source hashes | New current CPU evidence; cannot authenticate earlier cluster execution |
| Original Transformers receipts | `historical/C-trainer.private.json`, `historical/LQ-trainer.private.json` if available; exact expected raw hashes enforced | Local packet is partial if these are missing. Pasted summaries and Python-interface receipts are not substitutes |
| Seven missing source snapshots | `source/current/`; standalone verifier checks all seven requested digests | Recomputed byte identity, not historical consumption |
| Gates and collator/cache negative tests | Included existing `test_lexical_pair.py`, `test_csham_lq_feasibility.py`; targeted supplementary test result | Scoped CPU assertions only; no stored valid execution grant |
| Transfer/seen-context/utility | Unchanged proposal stage fields | 132 transfer attempts; 32 optional disabled; 24 utility episodes with unset step/call limits |

## Historical linkage and availability

The v3-specified original Jupyter receipt bytes were not found in the local checkout or attachment directory at preparation time. The expected hashes are retained in the index; the packet must be called **partial** until those exact files are supplied. Original local Python-interface receipts were not renamed or substituted. The packager accepts `--receipt-dir` and copies exact original bytes without rewriting or adding fields.

No contemporaneous run-local input manifest was recovered for the historical Jupyter checks. Even when both receipts are supplied, the new supplemental index establishes membership and current consistency, not which files the earlier process consumed. The new bound checks deliberately close the current input/code linkage gap only.

Native masks are existing saved evidence. Current checks do not rerun text tokenization. Real Transformers TrainingArguments/tensor construction is NOT RUN locally in this closure; reported earlier Jupyter results remain supplied-runtime claims until the original receipt bytes are included, and remain unauthenticated as cluster execution afterward. No real Trainer is constructed by the closure tool.

## Separate remaining prerequisites

- Candidate/CPU: deliver the exact original two receipts; review all eleven source assessments and unresolved wording/lineage. Missing bytes do not establish candidate failure.
- Training: scoped source/final-candidate decisions, initialization/weight identities, resolved optimizer/loss/trainable-module/precision/topology/sampler/update configuration, resources and separate installation/job/training grants. No framework default resolves a null optimizer.
- Transfer: frozen checkpoints, sixteen task/split/serializer/scorer bindings and distinct grants. Allocation remains 132; missing future tasks do not invalidate CPU construction.
- Seen contexts: four reviewed contexts and a separately allocated 32-attempt grant; disabled otherwise and never pooled with transfer.
- Utility: twelve paired marker-free tasks, absolute/degradation thresholds and model/environment episode budgets. Twenty-four episodes do not mean twenty-four responses.
- Later Shield/Seek: separate protocols and matching limitations; legacy semantic checkpoints remain unmatched.

All execution, training, model, resource, installation, job, source/stage/final-corpus/provenance/split/matched-control and method-matrix approvals remain false/not granted. C_sham is cue-exposed benign supervision; the contrast cannot isolate induction versus suppression relative to an untested starting model.

## Reproduce without running archived code

After inspecting the standalone verifier source, extract **only that script** from `source/current/scripts/seek/verify_csham_review_packet_v3.py`, then run it with standard-library Python against the packet. It reads archive members in memory, rejects unsafe/duplicate/symlink/missing members, verifies nested manifests and joins, and never imports archived candidate code. The final verification is also run in an isolated temporary directory with no repository on Python's import path.

To supply missing originals, copy the unmodified `C-trainer.private.json` and `LQ-trainer.private.json` from the Jupyter run directory `csham-lq-feasibility-ezmQXK`. Do not reconstruct them from terminal output. `CODEX_PROMPT.md` specifies their expected hashes. The companion `COMMANDS.md` contains exact packaging commands; no new GPU/native preprocessing run is needed.
