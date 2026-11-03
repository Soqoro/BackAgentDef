# Codex handoff v3: complete the C_sham/LQ candidate and CPU evidence package

**Date:** 2 November 2026.  
**Purpose:** Close the specific evidence gaps in the latest `insufficient_evidence` review before extending the experiments.  
**Status:** Development-checkout implementation, evidence packaging and bounded CPU testing only. This is not a completed independent review, source approval, final-corpus approval, installation permission, or model-execution grant.

This handoff supersedes the immediate work ordering in `Seek_CSham_LQ_Feasibility_v2_2026-11-02/CODEX_PROMPT.md`. Preserve that prompt and all prior artifacts. Retain its scientific design and later-stage boundaries unless explicitly clarified below. Do not invent a new marker, target, corpus, or training recipe to resolve an attachment problem.

## 1. Immediate objective and limits

The latest review received only two trainer-input receipts. It could verify their integrity, but could not inspect the expected proposal, eleven-source dossier, candidate records/annotations/arrays, or seven current implementation files. **Missing evidence is not a demonstrated implementation failure.**

First assemble an inspectable, correctly bound package from existing artifacts. Then run or implement only the specific CPU checks that are still missing. If the evidence and tests already exist and match the reviewed inputs, packaging them is a valid completion; code changes are not mandatory. Do not return another receipt-only submission, rebuild the framework, or repeat unchanged tokenization just to create a newer timestamp.

Permitted: read-only inspection and hashing of available assets; additive development source/configuration/test changes; private review-package construction; CPU parsing, saved-array checking, available native preprocessing when specifically needed, and isolated no-model integration tests. Keep legacy trainer defaults, Shield, `agent_eval.sh`, the base corpus, private marker/mapping, historical negatives and registry unchanged.

Not permitted: production installation or changing deployed entry points; model/weight loading for computation; real Trainer construction in this task; forward/backward passes; optimizer construction or steps; generation; training; GPU use; environment actions; dependency installation/downloads; paid calls; job submission; git push; approval/registry/stage mutation; or method-matrix execution. Do not author an independent approval for your own work. Non-executing inspection of model/configuration manifests is allowed.

Private corpus text, marker registries, substitution values and evaluator truth belong in the private reviewer packet, not Git or public summaries. Provide necessary private evidence to the authorized reviewer rather than replacing it with assertions. A blind Seek investigator may later see the actual public incident text, including a cue naturally present in it; do not supply its gold trigger identity, checkpoint condition, or private answer mapping.

## 2. Identify the correct proposal, receipts and earlier native evidence

Expected **current proposal content hash**:

`ad5716ed46224d759038c69477dc141954fa6da0b2316d9231de4b9ead3f9c8f`

Preserve these supplied receipts byte-for-byte:

| Receipt | Raw-file SHA-256 | Content seal (`hash`) |
|---|---|---|
| `C-trainer.private.json` | `924d0a63c5c6b91d1ee23521bcbbd9edf58d61b37856906053a3646c3a6caa22` | `b5ab27dea862043cc34d47833c61a8c5427ad1e5f5589a614dec4fbb76e58ee4` |
| `LQ-trainer.private.json` | `ca8e0fd78dc6f6eb71a27451e9db7bd97a7256496df7bb78fe454c024f508ffd` | `7501187a859dc7b4731d3c7419e3fe693c2f527348462e65ed61902da4937b9c` |

The earlier native export was reported under different identifiers:

| Earlier artifact | Identifier |
|---|---|
| Raw native-export ZIP | `6e368e2a0dc4039b8356615d6350ea05f038678f3ac2828e77510e11ba469697` |
| Raw outer manifest | `ebc19fcfa69a13f1f780b20ecccb5f16d4398fd11e37a54d4dfce1df87f3c7d6` |
| Native export content seal | `7cf3184038a669797b13ef6a752c7dabcc293403fea0f28164927dc94fdd1f15` |
| Earlier candidate plan content seal | `cacc4921e56f421024193f70c16b8503c9efff51d4031be172df59d70685a357` |
| Common native foundation content hash | `8b1165ac799dc4c2bd96e0a13a1c39a098fc73cbb9c8ab453026dc545ae1fed1` |

The current proposal and earlier plan are **different objects**. Do not equate them or assume a parent-child relationship from this table. Supply their actual bytes, explicit references and content comparisons establishing which corpus, annotation and array identities are shared. A proposal may legitimately refer to an earlier unchanged corpus; prove that relationship rather than requiring all object hashes to match.

Locate the exact expected proposal before interpreting its contents. If it cannot be recovered, mark it unavailable. Do not manufacture a file claiming the expected hash. A newly reconstructed or revised proposal has its own identity and cannot substitute silently.

Use each artifact's actual hash convention. Distinguish raw-file digest, canonical content seal, archive manifest and ZIP digest. A matching self-seal proves internal consistency, not upstream origin, cluster authenticity, or actual historical consumption.

## 3. Close the evidence gap without retroactively inventing provenance

Create a versioned **supplemental evidence index/manifest** linking the unchanged artifacts. Reuse the repository's manifest tooling. For every necessary item record path, raw digest/size, content seal and canonicalization where applicable, purpose, dependencies, and availability. Use relative paths that work after unpacking.

Separate these claims:

1. **Packaged-byte integrity:** the reviewer has the bytes and can recompute their hashes.
2. **Current cross-artifact consistency:** record IDs, data identities, annotations and arrays join correctly.
3. **Historical runtime linkage:** contemporaneous commands, resolved input hashes or run records show what an earlier process consumed.
4. **New explicitly bound CPU check:** a new limited run checks identified inputs with identified code now.

A supplemental manifest created now can establish package membership and current relationships. It cannot by itself establish claim 3. Look for existing run-local input manifests, command/configuration captures and source snapshots. Preserve and cite them if they exist.

If the old receipts lack adequate input linkage, keep that historical limitation. For a reviewer-checkable current result, perform only the necessary **new bound CPU trainer-input check** after the exact candidate and code are available. Its new receipt must identify proposal, per-role corpus/annotations, native-array export, serializer/tokenizer metadata, selected row IDs, actual code and resolved CPU configuration. Label its processor honestly as saved arrays when appropriate. This is a targeted closure run, not a repetition of unchanged full tokenization and not retrospective authentication of the old run.

Do not edit either old receipt to add fields or re-seal it as though those fields had been captured originally. If a new check fails, retain the failure and the old evidence separately.

## 4. Include the bytes needed for candidate and source review

Use one private self-contained packet, or explicitly indexed companion volumes if necessary. Include:

- The exact current proposal, its manifest and relevant prior review/plan documents.
- Both role corpora and annotation sidecars, or a lossless base-plus-additions representation from which both exact corpora can be reconstructed and their identities checked. Preserve the actual serialization/hash rules.
- The unchanged foundation and original annotation evidence, or a directly included referenced archive containing them. Do not rely on an unmounted remote path or an expired chat download.
- The eleven-source dossier: original task/prefix, selected action boundary, permitted instruction edit, benign action justification, private target-rule application, action availability, annotations, known dependencies and unresolved wording/lineage.
- Frozen private trigger/control definitions, target predicate and source/scorer tests.
- Complete native rows/arrays, row-to-record/annotation joins, assistant spans, generation prefixes, token receipts and tokenizer/template/source bindings used for their derivation. Shared data may be stored once with explicit role-specific reconstruction.
- The current CLI, adapter/collator, scorer, gating code, dependency helpers and tests needed to inspect the actual path.
- The unchanged receipts and any existing or newly required bound CPU integration evidence.

Do not include model weights. Tokenizer asset bytes may be included where available and permitted; otherwise identify them and state what cannot be independently retokenized. Do not ship unrelated private data or the entire historical repository unnecessarily. A list of hashes without the necessary source/data bytes does not satisfy a byte-level review.

Specifically recover the following seven snapshots at the digests declared in the receipts. Paths below are relative to the repository; the first four are under `agent-backdoor-attacks/AgentTuning/WebShop/`:

| Path | Expected SHA-256 |
|---|---|
| `seek/three_way/feasibility.py` | `455e7a7e9b78d9d81b3cfadb121584b12412fdbc1b4b9a0bcb3359bd49c73195` |
| `seek/three_way/lexical_pair.py` | `c3eb16a2744cf590254c4b5c4be452693434918964b9b97a78cfa140316c27ec` |
| `seek/three_way/lexical_pair_entry.py` | `97cbad1ead70c065f337b95fe472c8bee3d5dda64a11bf6d3b0a44e6528a4038` |
| `seek/three_way/lexical_v2.py` | `e71c5a68bbb19c3bd69366939d4171a68aaff823596dcdd075d28ee0b79028ac` |
| `seek_three_way.py` | `3dc14c1d9b4b366b64dc600c0b5c70d98a37025ee028c2867d1d8359f26868bd` |
| `seek_three_way_readiness.sh` | `86cc789a24914d4b7d4f041554e2301bebe7cc1ad27ee1c1b6874c09f389b46e` |
| `seek_three_way_train.sh` | `43bbc0e8b3a7aa6872985e4109ea2b51d297efea29ecb941bad1ad1989e4190e` |

These seven are the known missing subset, not a substitute for relevant transitive helper/test sources. If the checkout has changed, include separate old and current snapshots where recoverable. Never label current bytes with an old digest. Missing old snapshots remain missing; a new CPU result can support current code, not authenticate the old receipt's run.

Before returning the packet, validate archive paths, duplicate/missing members, file sizes/digests, and all required referenced files without executing untrusted archive code. Reopen the completed archive and run a reviewer-view completeness check from the unpacked bytes alone, with no undeclared access to the originating checkout. Verification tooling may be explicitly invoked after its own source inspection. Report missing evidence as missing, not pass. Do not create a circular manifest hash definition or promise a recursive ZIP digest inside itself.

## 5. Check the actual candidate; do not treat receipt counts as membership proof

The expected candidate is **1,858 unchanged foundation records + 44 additions per role = 1,902 each**, with eleven sources crossed with active/absent/decoy/near. These are expectations to verify from actual bytes, not instructions to force a match.

Check exact ordered membership, annotations, record joins, deliberate training reuse and C_sham/LQ alignment. Preserve all original source annotations and private definitions. Do not require 50 sources, automatically increase poison weight/epochs, remove base examples, or deem eleven sufficient from counts alone. These are proposed inspected training/development sources, never held-out tasks.

Assess each source separately. Retain unresolved requirements and earlier exclusions/ambiguities. A valid `search[...]` format is not proof of task suitability. A fixed next-search transformation need not have a same-page target product; click/purchase claims require real availability and transition evidence. Do not infer goals or execution from saved click text. Do not splice pages or fabricate continuations.

Verify the existing trigger predicate and designated region against actual consumed prefixes, including normalization, unintended occurrences and distinct active/control token inputs. Do not redefine X to fix a conflict. Check **within-role** incompatible-label collisions, including applicable foundation/action prefixes. Intended cross-role active-prefix equality with different continuations is the intervention, not a collision defect. Different acceptable benign phrasings are not automatically incompatible.

Run the frozen scorer against every authored label using the immutable original task and legitimate available actions. It must yield benign=0 and target=1 at the specified boundary without reading role, cue-state or filename. Record exact-match versus behavioral-rule scope as implemented; do not broaden the scorer to make labels pass. Malformed/ambiguous output is unknown, not automatically a non-target. Keep action validity, target match and goal conflict distinct. Negative fixtures should test metadata invariance and the rule's actual ambiguity cases.

Preserve base annotations. New additions retain only the original pre-action prefix, mask preceding assistant context and supervise the final action/terminator. No original final response or future observation may leak into the generation input. Check target-action coverage and equality of corresponding generation inputs; complete active training sequences correctly differ at their final continuations.

Use the existing full native evidence when its identities and scope genuinely match. If checking adapter outputs, feed the real processor's native labels/inputs and annotations through the adapter; do not merely reload final annotation-aware labels and compare them to themselves. Report source-text retokenization, saved-boundary replay, annotation application, and runtime collation as different checks.

Changes discovered to be necessary are proposed as new versioned corrections. Do not silently repair source text, mapping, membership or labels inside the supposedly unchanged candidate. No source is independently approved by Codex's own dossier.

## 6. Verify the CPU integration boundary precisely

The two receipts report 1,902 constructed records and **45 collated rows per role: one base row plus 44 additions**, using saved arrays. They report no model load or Trainer construction, and an unresolved optimizer. Preserve that exact scope.

Distinguish dataset length/construction, per-record materialization/validation, and rows actually passed through the collator. Earlier complete native-mask coverage plus a properly bound representative integration check can be useful. Do not require another full 1,902-row run merely because 45 is a subset, and do not describe 45 as full-dataset collation. Add edge cases only where actual path/coverage needs justify them.

Inspect and document the production-intended opt-in path:

`CLI/config -> AnnotationDataset -> guarded collator -> TrainingArguments configuration -> Trainer call site`

There are three evidence levels, none interchangeable:

- **Source trace:** exact files/line ranges show the path and supplied arguments.
- **Isolated CPU wiring test:** exercise the real entry/config/dataset/collator path with controlled constructor/factory spies at the actual call sites. Capture would-be arguments and label/data identities. Do not use a parallel test-only reimplementation of the path.
- **Real Trainer/model behavior:** not exercised or approved here. A constructor spy is not a real Trainer instance, loss computation, optimizer, resource test or deployed integration proof.

A real TrainingArguments/config object may be constructed only after inspecting the pinned path and ensuring it stays CPU-only and cannot initialize devices/distributed services or load a model. Otherwise inspect/serialize its exact intended inputs and use an explicitly labeled test double. Missing installed dependencies are NOT RUN, not simulated native success. No real Trainer instantiation or dummy-model forward/backward is required or permitted to satisfy this handoff.

Use existing tests where valid. Close only uncovered assertions: label-preserving collation (including a deliberately overwriting negative fixture), cache isolation including role-specific targets/annotations, prefix equality, attention/padding, causal-shift ownership, and applicable annotation-boundary fixtures. Bound tests to their actual imported sources. Do not infer correct loss normalization from equal token totals.

Test absent/mismatched grants and dry-run behavior with sentinels placed before model loading, devices, training, generation, jobs and environment calls. If authorized-branch argument assembly needs simulation, use an isolated in-memory test double, never a persisted valid-looking execution grant, production bypass flag or modified registry. Label this as mocked wiring. Passing a false flag in a receipt is not a gate test.

If a concrete defect is found, report it with a minimal reproducible test, patch additively, preserve the old snapshot/receipt and issue a new scoped test result. Do not declare an uninspected guard broken, and do not install changes into the production workflow.

## 7. Prepare staged prerequisites, not a new all-or-nothing gate

Use a requirement-to-evidence table with explicit states such as packaged/recomputed, source-inspected, supplied-runtime-only, mocked-wiring, NOT RUN, unavailable, or defect-found. Each row names exact files/fields/line ranges, reviewer reproducibility limits and the stage it blocks. These are evidence states, not self-issued approvals.

| Stage | Prerequisites to keep separate |
|---|---|
| Candidate/CPU review now | Actual proposal/data/source bytes; eleven-source dossier; consistency and scope-correct integration evidence. A positive model effect is irrelevant. |
| Training feasibility | Scoped source/final-candidate decisions; initialization and weight identities; opt-in integration; fixed optimizer, trainable modules, precision/topology, sampler/update/batching/loss convention, checkpoint-selection recipe; resources and separate installation/job/training grants. |
| Transfer inference | Frozen trained checkpoints; reviewed sixteen-task population and held-out axis; actual serializer/scorer; the unchanged 132-attempt protocol and distinct inference/resource grants. |
| Optional seen-context inference | Four prospectively selected training contexts; separate 32-attempt allocation/grant; disabled otherwise; results not pooled with transfer. |
| Utility | Twelve paired marker-free tasks, absolute usability and degradation criteria, full-episode call/action limits and environment/inference/resource grants. |
| Later Shield/Seek methods | Separate protocols, valid attack/utility scope, matching limitations and authorization. |

Unresolved optimizer, initialization or resource fields do not automatically invalidate bounded CPU input evidence. They remain blockers to training, and must not silently resolve to framework defaults. Record available values with their provenance; mark proposed or missing values explicitly. Do not invent a memory fit, scheduler, full-tuning/LoRA choice, seed robustness, loss convention or matched-checkpoint status.

Missing future transfer tasks are not defects in current CPU construction. They prevent transfer claims/execution, not independent packaging and source tests. Keep a restricted fitting-only proposal distinct; do not waive source/corpus/training requirements or make utility claims from it.

## 8. Preserve the experimental design and budgets

Keep C_sham as a cue-exposed benign-label control. The contrast does not isolate LQ induction versus C_sham suppression relative to an untested starting policy. One realized pair is not training-seed replication. Legacy SQ/SO remain unmatched; different targets/interfaces cannot isolate trigger-form difficulty.

Do not alter current allocations:

- Transfer/readiness: **16 tasks x 4 states x 2 roles = 128 responses + 4 anchors = 132 maximum attempts**, no hidden retries or warm-up generations.
- Optional seen-training-context diagnostic: **4 x 4 x 2 = 32 attempts**, disabled unless separately allocated/approved. Never pooled with transfer or silently included in 132.
- Benign utility: **12 paired tasks x 2 models = 24 episodes**, with separate response/environment limits; not 24 responses.

Retain signed active-minus-control outcomes and excess over sham. Preserve measurement validity, primary active/absent dependency, excess, exact-marker profile, broader cue response and utility as separate statuses. Active/absent alone cannot establish exact lexical specificity. Near-cue activation is not exact-marker success. Seen decoy/near strings on new tasks do not demonstrate arbitrary unseen-string specificity.

Carry forward proposed thresholds and failure policies without silently amending them. Logical intervals for unknown outcomes are not statistical confidence intervals. No pilot results, theorem certificates or positive effects exist in these CPU receipts. No model must be run to complete the present review package.

## 9. Return a complete, bounded submission

Return:

1. One private complete candidate/CPU review packet (or a clearly indexed multipart set), with its new raw archive/manifest digests and unchanged proposal identity where verified.
2. A concise `REVIEW.md` mapping every outstanding requirement to actual files and evidence types, with a direct note on historical versus retrospectively assembled links.
3. The unchanged two receipts, source snapshots at their stated digests, and only necessary newly bound CPU tests/receipts. Separate old and current source identities after any changes.
4. Eleven individually documented source assessments and preserved unresolved issues; no independent approvals inferred from consistency tests.
5. A short current-candidate/CPU status report and separately grouped training/transfer/seen-context/utility prerequisites. No requirement to complete future experiments now.
6. Exact existing or implemented CPU commands, test results actually obtained, and precise missing-input instructions. Do not invent CLI commands or successful runs.

Reopen and validate the final package before delivery. If a necessary item remains missing, mark the packet partial and identify the exact missing bytes; do not advertise it as a complete review submission.

All training, execution/inference, model-execution, resource, production-installation, job, stage-assignment, final-corpus, provenance, split-independence, matched-control and method-matrix approvals remain false/not granted. Leave reviewer identity/decision unfilled in any template; do not fill in `me` or manufacture a signature. Preserve prior decisions unchanged.

**Completion criterion:** The reviewer can inspect the actual candidate and intended CPU path, or can see precisely which evidence is still unavailable. Packaging alone is sufficient when that is the only gap. Targeted CPU work closes demonstrated gaps; new model experiments and another attack redesign are not this task.
