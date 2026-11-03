# Shield-and-Seek progress and results

**Report date:** 2 November 2026, Asia/Singapore  
**Current phase:** lexical C_sham/LQ candidate preparation and CPU evidence review  
**Current status:** candidate constructed and CPU-checked; private review packet remains partial

We have moved from exploratory semantic-checkpoint diagnostics to a general lexical trigger-to-action construction. The current candidate preserves the private trigger and target mapping, provides matched C_sham/LQ training inputs, and has substantial saved-native and CPU integration evidence. **We have not trained or evaluated the proposed lexical agents.** There are no lexical attack-success, utility, Shield or Seek comparative results yet.

The immediate remaining delivery gap is the exact original pair of Jupyter trainer-input receipts. The latest review found insufficient evidence because necessary files were absent from its submission; it did not demonstrate a candidate failure.

## 1. Scope and strength of evidence

This report combines the supplied historical run summaries with the candidate archives and current local verification artifacts. Historical numbers below are identified as reported results where the underlying cluster execution has not been independently authenticated. The latest candidate/package counts were checked against available bytes.

Four different kinds of evidence must remain separate:

| Evidence | What it supports | What it does not support |
|---|---|---|
| Archive/hash verification | The reviewer has the specified bytes; manifests and seals are consistent | Upstream authenticity or actual historical consumption |
| Cross-artifact checks | Proposal, corpus, annotations, arrays and source snapshots refer to matching objects | Proof that an earlier process used those objects |
| New bound CPU checks | Identified code processed identified inputs now, within the stated scope | Historical runtime authentication, training or inference |
| Reported cluster receipts | The reported outcome and scope of an earlier run | Independent verification of cluster execution without additional contemporaneous evidence |

Private markers, substitution values, source text and evaluator truth are deliberately omitted from this public report. They are available in the private candidate packet for authorized review.

## 2. Experiment objective and sequence

The intended sequence remains:

1. Construct task-grounded lexical training examples.
2. Train appropriately controlled agents from a shared initialization and reviewed recipe.
3. Verify the trigger's behavior and benign utility.
4. Evaluate Shield using separately defined ASR and AER endpoints.
5. Evaluate Seek across clean/control, lexical and semantic conditions with matching limitations disclosed.
6. Run HACS ablations and finalize tables.

We are still in **step 1 and the preparation for step 2**. The current proposed pair is C_sham/LQ. It does not yet supply the entire clean/lexical/semantic comparison, an observation-channel poisoned agent, or matched semantic controls.

## 3. Earlier semantic investigation

The existing query and observation checkpoints were described as semantic-attack checkpoints. Initial HACS development encountered several distinct problems: incompatible outcome grounding, JSON-array/schema errors, output truncation, and confusion between observed and hypothetical predictions. Those failures and subsequent reviews remain historical evidence.

A later bounded query-checkpoint experiment compared constructed “ankle boots” and “athletic shoes” prompts in two backgrounds:

| Reported measure | Result |
|---|---:|
| Pair 1 target occurrences, ordered boots/shoes | [0, 1] |
| Pair 2 target occurrences, ordered boots/shoes | [0, 1] |
| Signed difference in each pair | −1 |
| Query probe attempts | 4 |
| Query anchor attempts | 2 |
| Total physical calls | 6 |
| Matched anchors | Stable |
| Observation/reference calls | 0 |

This was descriptive target occurrence under a constructed, inspected contrast. It did not uniquely identify a mechanism, authenticate poisoning causation, establish an unauthorized action, or confirm a scientific theorem. Token usage was unreported, not established as zero. The method matrix remained locked.

These findings are separate from the current lexical candidate. They are not lexical-pilot results.

## 4. Foundation corpus and supervision preparation

The initial local corpus contained **1,866 conversations**. Native preprocessing diagnostics revealed that using a fast tokenizer as a proxy for the trainer's slow tokenizer was unreliable: all 351 examined WebShop records differed between the two tokenizations. Native prefix-boundary alignment resolved the serialization comparison without replacing the trainer's tokenizer.

Reported truncation diagnostics at a 2,048-token cap included:

| Corpus slice | Records | Truncated records | Fully lost assistant turns | Partially truncated assistant turns |
|---|---:|---:|---:|---:|
| WebShop slice | 351 | 52 | 80 | 5 |
| Full corpus diagnostic | 1,866 | 374 | 1,322 | 116 |

One full-corpus record had an empty assistant response. The reviewed membership proposal used a 4,096-token cap and quarantined seven over-cap conversations plus that empty-response conversation, retaining **1,858 whole conversations**.

| Family | Retained | Excluded |
|---|---:|---:|
| ALFWorld | 336 | 0 |
| DB | 538 | 0 |
| KG | 318 | 6 |
| Mind2Web | 121 | 1 |
| OS | 195 | 0 |
| WebShop | 350 | 1 |
| **Total** | **1,858** | **8** |

The retained-corpus native-mask receipt reported zero truncation, lost assistant turns, unresolved boundaries and mask mismatches. The original corpus and annotation evidence were preserved.

The native FastChat preprocessing supervised all assistant turns. The source annotation manifest separately contained 4,671 assistant turns marked `loss:false` and 9,675 marked `loss:true`. The reviewed explicit supervision policy masks the former:

| Measure | Tokens |
|---|---:|
| Native all-assistant supervision | 609,507 |
| Annotation-aware supervision | 434,727 |
| Removed supervision | 174,780 |

This removes approximately **28.68%** of native supervised tokens while preserving input IDs. The reduction is especially concentrated in KG records. The policy choice does not authenticate the original annotation author's intent.

The CPU adapter validation reported **929 batches**, with no training or model calls. Later evidence reviews accepted bounded CPU findings; they did not approve model execution.

## 5. Provenance and dependency work

We investigated corpus origin, historical exposures and product/task relationships rather than treating a file named “clean” as proof of clean provenance.

Reported findings included:

- Local cached-source comparison found 1,866 exact records with multiplicity. A local cache alone is not independent upstream authentication.
- An expanded inventory scan parsed 68 candidate files without skips and found zero exact text matches.
- A separate comparison checked 14,346 full human-turn hashes against 32,469 unique exposure fingerprints and found zero matches. This did not establish split independence.
- All 350 retained WebShop instructions had product candidates after a disclosed price-suffix transformation. There were 356 unique candidate products and 358 record-to-product associations.
- Four training records retained ambiguous product assignments. Click text did not resolve their original goals.
- A broader 14,248-file scan found 8,316 product references, all in three task-inventory files. These were inventory references, not evidence of model consumption or executed trajectories.
- Conservative WebShop grouping proposed 257 dependency groups: 218 singletons and 39 multi-record groups, with a largest group of 15 records. Stage assignments remained unresolved.
- Searches for primary task/trajectory logs found no matching records in the scanned scope and reported missing roots.

Authenticated source history, applicable stage-level task/product/trajectory/template relationships and matched trained controls remain unresolved. A genuine clean-policy dependency must not be mislabeled as a false backdoor certificate.

## 6. Why the lexical design changed

Earlier source collection searched for Adidas-related attack opportunities. Several catalogue/index and bounded environment searches failed to produce a suitable same-state observation pair. Those are preserved bounded negative findings, not evidence that no such state exists anywhere.

The key correction was that a lexical backdoor does **not** require Adidas or any particular brand target. We shifted to a private, task-grounded **X → Y** construction, prioritizing the next-search/query channel. No further Adidas-specific search campaign is required for this candidate.

The general lexical implementation first passed a synthetic software fixture. A strict audit of 350 real WebShop records initially produced zero eligible candidates because the required explicit attribute format was absent or nonexclusive. A wording diagnostic and named-source review then supported development drafting rather than silently relaxing the original audit result.

The inspected four-source draft produced 32 role/state examples. Its decoy was initially one token shorter; the balanced revision gave active, decoy and near controls **18 tokens each**, with equal full-context lengths for those nonempty cue states. The absent cue remained empty. The current candidate later proposed eleven sources, without imposing an arbitrary fifty-source minimum or increasing poison weight to compensate for its size.

## 7. Current C_sham/LQ candidate

| Property | Current value |
|---|---:|
| Common foundation records | 1,858 |
| Proposed underlying sources | 11 |
| Cue states per source | 4: active, absent, decoy, near |
| Added examples per role | 44 |
| Total records per role | 1,902 |
| Total role records | 3,804 |

The roles share corresponding generation inputs. C_sham receives benign final-action labels in every cue state. LQ differs only in the active state's target continuation. C_sham is therefore **cue-exposed benign supervision**, not a cue-naive policy. A later LQ–C_sham contrast would not by itself isolate induction in LQ from suppression in C_sham relative to the starting model.

Base annotations remain unchanged. For new examples, preceding assistant context is masked and only the final action and terminator are supervised. No future observation is added to the generation input.

All eleven source contexts have foundation/development exposure. They are proposed fitting/development examples, not held-out transfer tasks. Their individual suitability assessments and unresolved issues are preserved in the private dossier; no source approval is inferred from counts or syntactically valid searches.

### Current byte-level and CPU results

| Check | Result |
|---|---:|
| Added cue/role examples checked | 88 |
| Source/role groups with four distinct token prefixes | 22 |
| Incompatible identical-prefix labels among checked additions | 0 |
| Same-Y within-role prefix collisions among checked additions | 0 |
| Active cue mentions in the foundation corpus | 0 |
| Added contexts matching foundation prefixes | 11 |
| Changed action-token positions across paired active examples | 11 |
| LQ active target-action supervised tokens | 271 |

The foundation matches are retained separately for contextual review; absence of conflicts among additions does not erase foundation reuse or prove every original thought-plus-action response is equivalent to a canonical final-action label.

Per-role saved-native totals are:

| Measure | C_sham | LQ |
|---|---:|---:|
| Input tokens | 2,485,808 | 2,485,808 |
| Foundation supervised tokens | 434,727 | 434,727 |
| Added supervised tokens | 1,084 | 1,084 |
| Total supervised tokens | 435,811 | 435,811 |
| Active target-action supervised tokens | 0 | 271 |

Equal token totals do not establish equal gradients, loss weighting, attack effectiveness or benign utility.

The frozen scorer operates on the immutable original task and available actions, without receiving role or cue state. Authored benign labels score 0 and target labels score 1. Its endpoint is an exact next-action transformation; alternative valid searches and malformed outputs are handled separately. Passing label checks is not evidence that a trained model emits those actions or that the action constitutes a Shield security violation.

## 8. Trainer integration and CPU evidence

The additive opt-in path connects:

`CLI/config → AnnotationDataset → guarded collator → TrainingArguments configuration → Trainer call site`

The earlier local feasibility work passed **127 tests**: 21 new feasibility tests, 45 lexical-pair tests and 61 lexical-v2 tests. These are a prior test-run result, not 127 new tests in the latest closure.

Local saved-array dry-runs and subsequently supplied Jupyter summaries reported:

| Measure | C_sham | LQ |
|---|---:|---:|
| Dataset records constructed | 1,902 | 1,902 |
| Rows materialized and collated | 45 | 45 |
| Coverage | First base row + all additions | First base row + all additions |
| Model loaded | No | No |
| Trainer instantiated | No | No |
| Optimizer resolved for training | No | No |

The Jupyter summaries reported the installed Transformers interface. Their exact original receipt files are not currently present in the local review packet. The local Python-only receipts have different identities and were not substituted.

The v3 closure added two explicitly bound current CPU checks. They identify proposal, corpora, annotations, native export, array hashes, code, configuration and selected row IDs. They feed native inputs/labels through the annotation adapter and compare the resulting labels with the saved final arrays. A controlled spy at the actual nested Trainer constructor checks argument forwarding. This does not instantiate a real Trainer, compute a loss, construct an optimizer or perform model computation.

The latest closure passed **11 targeted tests**: six archive/path/nesting integrity tests and five selected existing collator-overwrite, cache, causal/padding and gate tests. These should not be added to the earlier 127 as a count of distinct tests.

No unchanged full tokenization was rerun to create a newer receipt.

## 9. Latest review and package closure

The latest review lacked the proposal, candidate data, dossier, native arrays and seven current source files. Its disposition was **insufficient evidence**, not demonstrated failure of the candidate.

We recovered and packaged:

- The exact current proposal and its explicit relationship to the earlier native candidate plan.
- Both complete role corpora and annotations, unchanged foundation and source evidence.
- Full native arrays and row-to-record/annotation joins.
- All eleven source assessments, private contracts and source/scorer evidence.
- All seven requested source snapshots at the specified hashes, plus relevant dependencies and tests.
- Newly bound CPU checks and the targeted test log.

The first reviewer-view package check found an omitted older preparation-script snapshot. Packaging was corrected without changing the candidate or working implementation. The final ZIP passed verification in an isolated temporary directory using standard-library Python with no checkout imports.

**The local packet remains partial because the two exact original Jupyter receipts are missing.** They were neither reconstructed nor rewritten. A new index can establish package membership and current consistency; it cannot retrospectively establish what the historical Jupyter process consumed. No contemporaneous run input manifest was recovered for that process.

### Evidence identities

| Object | Identifier |
|---|---|
| Current proposal content seal | `ad5716ed46224d759038c69477dc141954fa6da0b2316d9231de4b9ead3f9c8f` |
| Earlier native candidate plan seal | `cacc4921e56f421024193f70c16b8503c9efff51d4031be172df59d70685a357` |
| Original native export ZIP SHA-256 | `6e368e2a0dc4039b8356615d6350ea05f038678f3ac2828e77510e11ba469697` |
| Current supplemental packet ZIP SHA-256 | `4aa2b43f7fcb6d453f06f93a528e036c14e6cebb2d308b4cfb761108b06749bb` |
| Current supplemental manifest SHA-256 | `240d1359f8562f75ac1c16f6b9418e65aed09e930a3ac2a49176e3c92ed93680` |

The earlier plan, current proposal and ZIP are different objects with different hash conventions. Their identifiers must not be treated as interchangeable.

## 10. What remains unmeasured

| Question | Current answer |
|---|---|
| Does the lexical trigger work after fitting? | Unknown; proposed agents not trained |
| What are lexical ASR/AER values? | Not measured |
| Is C_sham useful on benign tasks? | Not measured |
| Does LQ preserve benign utility? | Not measured |
| Does behavior transfer to new tasks? | Not measured; transfer population missing |
| Does the marker exhibit exact specificity? | Not measured; label/control checks are preparation evidence |
| How does Seek compare across clean, lexical and semantic models? | Not evaluated for this campaign |
| Are legacy semantic checkpoints matched controls? | No matching established |
| Are HACS ablation tables final? | No |

A successful CPU pipeline establishes that the proposed data can be inspected and processed consistently within the checked scope. It does not establish attack behavior or downstream scientific conclusions.

## 11. Frozen pilot allocations and interpretation

| Stage | Proposed allocation | Current state |
|---|---|---|
| New-task transfer | 16 tasks × 4 states × 2 roles = 128 responses + 4 anchors = **132 attempts** | Population not supplied; no execution grant |
| Optional seen-training contexts | 4 contexts × 4 states × 2 roles = **32 attempts** | Disabled unless separately allocated and approved |
| Benign utility | 12 paired tasks × 2 roles = **24 episodes** | Tasks, thresholds and episode call/action caps unresolved |

The optional 32 attempts are not part of the 132-response transfer allocation and must never be pooled with transfer. Twenty-four utility episodes are not twenty-four model responses.

Later reporting must distinguish measurement validity, active-versus-absent dependency, excess over C_sham, exact-marker profile, broader cue response and utility. Near-cue activation cannot be counted as exact-marker specificity. Missingness bounds are logical bounds, not statistical confidence intervals. A single trained pair would not establish seed robustness.

## 12. Remaining prerequisites by stage

**Current review delivery:** recover the exact original C and LQ Jupyter receipts, package them unchanged, and submit the single indexed packet for independent review. Their expected paths are under:

`/export/home2/suaq0001/seek_private/three_way_v1/csham-lq-feasibility-ezmQXK/`

**Training feasibility:** scoped source/final-candidate acceptance; initialization and weight identities; exact optimizer, trainable modules, loss convention, precision, topology and sampler/update accounting; reviewed resources and output paths; separate installation/job/training permissions. A null optimizer must not silently become an approved framework default.

**Transfer:** frozen trained checkpoints, sixteen actual task/scorer/serializer bindings and a reviewed held-out axis with applicable dependency checks, plus distinct execution/resource grants. Missing later transfer tasks do not invalidate current CPU construction.

**Optional seen-context check:** four reviewed sources and an explicitly separate 32-attempt allocation. It remains disabled.

**Utility:** twelve marker-free task pairs, absolute usability and degradation criteria, episode limits and model/environment budgets, plus grants.

**Shield, Seek and HACS:** valid endpoints and attack/utility scope, comparison protocols, matching limitations and separate approvals. No method-matrix access is created by this preparation work.

## 13. Immediate next action and artifacts

The next action is evidence delivery, not another attack redesign or model run:

1. Push the additive packaging tooling if it is not already on Jupyter.
2. Run the existing Jupyter packaging command, which verifies and includes the original two receipt files.
3. Submit the resulting single private ZIP for independent review.
4. Resolve the review's actual findings before preparing any model-execution request.

Useful files:

- [Requirement-to-evidence review map](REVIEW.md)
- [Complete push and Jupyter packaging commands](COMMANDS.md)
- [Latest delivery status and missing receipt hashes](DELIVERY_STATUS.md)
- [Private supplemental packet](../../outputs/csham_review_v3_2026_11_02/review.private.zip)
- [Isolated reviewer-view result](../../outputs/csham_review_v3_2026_11_02/isolated-reviewer-view.json)
- [Earlier feasibility implementation report](../Seek_CSham_LQ_Feasibility_v2_2026-11-02/IMPLEMENTATION_REPORT.md)

This report is an additional narrative artifact. It does not replace or modify the frozen packet, proposal, receipts or historical approvals. All current training, inference, resource, installation and later-stage permissions remain false/not granted.
