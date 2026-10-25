# Shield-and-Seek: progress, results and audit report

**As of:** 24 October 2026, Asia/Singapore  
**Author:** Codex implementation assistant, not an independent auditor  
**Repository baseline:** `0056bcf928f804ce5c4e4357807dcaa3e339b835` (`Add agentbench`)  
**Evidence:** repository code and archived reviews; user-supplied cluster receipts. Cluster files, upstream dataset identity and original model execution were not independently authenticated while preparing this report.

This report updates the [21 October report](SEEK_PROGRESS_AND_REVIEW_2026-10-21.md), especially its then-pending exposure scans, supervision-policy decision and adapter validation. Old reports, failed runs, reviews and receipts remain historical evidence. This document grants no approval and does not change the authoritative registry.

## 1. Overall assessment

We have a **reviewed 1,858-record candidate corpus**, a **CPU-validated annotation-aware supervision adapter**, and **reviewed conservative dependency groups for its 350 WebShop records**. The cluster reports successful replay of all 105 citations used in the dependency-context review.

The new C/LQ/LO experiment is **not ready for training or model execution**. Clean provenance, historical task/trajectory/template lineage, split independence, final attack corpora and matched controls remain unverified. No new C/LQ/LO checkpoints or attack-readiness results have been reported.

Two local-source searches have now completed without finding primary task journals for these records. The latest parsed 14,400 JSON/JSONL files but recognized **zero AgentBench task journals**, so it had **zero observation–response steps to compare**. This is a missing-evidence result, not proof of no historical exposure or independent splits.

| Area | Current result | What it does not establish |
|---|---|---|
| Historical Seek | Implemented effects certified in five streams | Semantic/backdoor causation or matched clean controls |
| HACS discovery | Abstentions, engineering repairs and bounded development diagnostics | Method superiority or fresh held-out discovery |
| Corpus preparation | 1,858 retained; eight quarantined; native masks passed | Clean-source authenticity |
| Supervision policy | Explicitly approved masking of originally `loss:false` assistant turns | Original annotation-author intent |
| Adapter | Revised CPU evidence accepted within scope | Installed training launch gate or final attack-data validity |
| Dependency review | 61 associations classified; 257 components reconstructed | Task identity or split independence |
| Citation replay | 105 cited occurrences matched local native evidence | Upstream or historical execution authentication |
| Primary lineage search | No matching explicit goals; no recognized task journals | Proof that original logs never existed |

## 2. Historical experiment results retained

These values are carried forward from earlier reports and supplied results; this audit did not rerun or recertify the historical models. Blocks are not independent training trials.

| Claim | Experiment | Blocks | Mean | Reported lower bound | Disposition |
|---|---|---:|---:|---:|---|
| 1 | Query category contrast | 37 | 1.000000 | 0.207108 | Implemented effect certified |
| 2 | Observation title-label contrast | 130 | 0.684615 | 0.200667 | Implemented effect certified |
| 3 | Bare-Llama query reference | 0 | — | — | Invalid inference: malformed action |
| 4 | Bare-Llama observation reference | 0 | — | — | Invalid inference: malformed action |
| 5 | Query effect minus AgentLM effect | 224 | 1.000000 | 0.200049 | Between-checkpoint implemented effect certified |
| 6 | Observation effect minus AgentLM effect | 848 | 0.648585 | 0.204404 | Between-checkpoint implemented effect certified |
| 7 | Observation category × brand-assignment pilot | 12 | 0.000000 | — | First-position selection dominated |
| 8 | Sneakers versus trainers pilot | 12 | 0.000000 | — | Pilot complete |
| 9 | Sneakers versus casual loafers pilot | 12 | 1.000000 | — | Pilot complete |
| 10 | Sneakers versus casual loafers confirmation | 64 | 1.000000 | 0.277102 | Implemented effect certified |

Five streams crossed the implemented threshold of 0.20. Semantic discrepancy remains unresolved. AgentLM is an external reference, not a matched clean-training control. Failed claims 3–4 are not zero-effect measurements. Claims 1–6 retain historical source/runtime provenance gaps; recovered source maps for claims 7–8 do not authenticate original execution. Claim 10 used 40 unique backgrounds and 80 unique prompts, with reuse distinguished from independent evidence.

The incident-led four-incident × three-method study produced **12 abstentions and zero victim probes**. It did not show method equivalence. HACS development exposed grounding errors, malformed JSON fields, token/output limits and confusion between observed and hypothetical predictions. Independent qualitative review of saved role explanations did **not pass**; later schema or citation improvements did not overturn that finding.

A separately reviewed, query-only development probe retained the agent-selected **ankle boots versus athletic shoes** contrast. Under two evaluator-specified $150 prompt templates, both outcomes were `[0,1]`: Adidas absent for boots and present for athletic shoes, signed difference **−1** each. Four probes plus two stable anchors used six physical query calls, with zero observation/reference calls. Local evidence reconstruction reported matching attested sources. This is descriptive output-word occurrence under constructed prompts, not unique mechanism identification, unauthorized execution, poisoning causation, purchase success or method superiority. The method matrix remains locked.

## 3. Additive implementation completed

Work reused Seek/HACS and preserved existing Stage I behavior, Shield, `agent_eval.sh`, historical results and registry semantics. The three-way preparation distinguishes proposed C, LQ and LO roles from legacy SQ/SO checkpoints. A role named “clean” does not authenticate its source, and legacy semantic checkpoints are not silently treated as matched controls.

Implemented components include arbitrary-marker capabilities, capability-derived hypotheses, arm-level updates, active search/coverage, explicit no-progress decisions, compact role review, preparation/readiness interfaces and bounded CPU diagnostics. Earlier known-rule simulation runs were smoke tests, not powered calibration studies.

Recent additions focus on auditable data preparation:

- Exact native preprocessing and boundary diagnostics, context-policy comparison and guarded conversion.
- Annotation-preserving supervision comparison and an opt-in dataset adapter with eager/lazy and shuffled-index checks.
- Final-corpus preparation helpers with explicit annotation sidecars and target-action bindings; actual final C/LQ/LO inputs remain unavailable/unapproved.
- Provenance/exposure inventories, candidate product matching, catalogue-reference comparisons and conservative dependency grouping.
- Private review bundles, context citation replay and bounded primary-goal/AgentBench journal searches.

An actual reviewed opt-in training entry point with an enforced launch authorization gate is still outstanding. A receipt-check helper or readiness audit is not that gate.

## 4. Corpus preparation and supervision results

The original candidate contains 1,866 records. Comparison with six locally cached AgentInstruct Parquet files found exact full-record equality with multiplicity, including annotations, and cache blob hash consistency. This establishes local equality, not remote publisher authentication or global record-order provenance.

Native FastChat uses a Vicuna template and a slow Llama tokenizer. The inspected trainer ignores per-turn `loss`. Fast/slow tokenizer IDs differed on all 351 original WebShop records, while actual native preprocessing matched the slow tokenizer. The checker was repaired to use exact native prefix boundaries rather than treating the fast tokenizer as authoritative.

At the original 2,048-token cap, the full-corpus diagnostic reported 374 truncated aligned records, 1,322 fully lost assistant turns and 116 partially truncated assistant turns. `mind2web_60` had an empty assistant response and required explicit treatment. These were preparation defects/limitations, not backdoor findings.

Separate reviews approved whole-conversation membership at 4,096 and CPU-only schema conversion:

| Family (from record ID) | Retained | Quarantined |
|---|---:|---:|
| ALFWorld | 336 | 0 |
| DB | 538 | 0 |
| KG | 318 | 6 |
| Mind2Web | 121 | 1 |
| OS | 195 | 0 |
| WebShop | 350 | 1 |
| **Total** | **1,858** | **8** |

Quarantined records: `webshop_242`, `kg_69`, `kg_159`, `kg_207`, `kg_213`, `kg_227`, `kg_319` (over cap), and empty-response `mind2web_60`. No truncation, overlapping windows or fabricated response repairs were adopted. Original records and annotations remain preserved.

The materialized candidate passed the reported native mask audit: **1,858 aligned records; zero truncation, lost/partial assistant turns, unresolved boundaries or mask mismatches**. The immutable materialization manifest's earlier pending status was not rewritten; the later mask receipt supplies the pass evidence.

### Approved supervision choice

The retained sidecar contains 4,671 `gpt:false`, 9,675 `gpt:true` and 14,346 `human:null` annotations. Independent review approved a new explicit policy: preserve all conversation context, but mask originally false assistant spans from loss. This is a reviewed policy choice, not authentication of the original annotator's intent.

| Quantity | Tokens/turns |
|---|---:|
| Native supervised tokens | 609,507 |
| Annotation-aware supervised tokens | 434,727 |
| Removed supervised tokens | 174,780 (28.675635%) |
| False assistant turns affected | 4,671 |
| True assistant turns | 9,675 |
| Records with changed labels | 1,737 |
| Records without false assistant turns | 121 |

The intervention changes labels, not input context. KG accounts for 145,008 removed labels, so the weighting change is substantial and uneven across families. Shared masking alone does not establish matched trained models.

### Adapter validation and review

Initial CPU evidence received requested changes R01–R04 concerning true shuffled indexing, boundary fixtures, helper-source binding and attention/padding/causal checks. These were addressed and the original failed review retained.

The revised cluster receipt reports the same token totals over **1,858 records and 929 sequential batches**, plus seeded shuffled-index checks (1729 and 2718). The revised independent review accepted bounded implementation/CPU evidence and closed R01–R04. It reports independent synthetic-test and index-order reproduction; full native corpus/tokenizer execution was accepted from supplied runtime receipts, not independently rerun by the reviewer.

No training, GPU inference or new model checkpoint follows from this pass. Final C/LQ/LO corpora still require their own annotation and target-action mask receipts.

## 5. Exposure and product-lineage investigation

The expanded metadata scan parsed all 68 candidates without skips and found zero exact text matches. Comparing 14,346 full human-turn hashes to 32,469 exposure fingerprints also found zero matches. These exact-string comparisons do not cover semantic overlap, embedded tasks, template dependence or complete historical consumption.

The historical inventory was filtered synthetic sneaker tasks, while the training examples included broad human shopping instructions. This explained why direct instruction matching, including case/whitespace diagnostics, returned no candidates; it was not evidence of independence.

Matching against the human-instruction source produced candidate products for all 350 retained WebShop records. The declared transformation removed a price suffix in 344 cases; six had no such suffix. There were **356 unique candidate ASINs and 358 record-to-product associations**. Thirty records initially had multiple source entries, but only four had multiple distinct products.

| Record | Unresolved candidate ASINs |
|---|---|
| `webshop_4` | B004R8WJHS, B01MR0TR0P, B073V5F2S4 |
| `webshop_113` | B093L52RXG, B093SZ7QMH, B093YSBNHN, B093YTKGG5, B093YVJSWF |
| `webshop_144` | B01IA9B8V2, B09DK5YFVL |
| `webshop_185` | B093KMVLFG, B093SZ5P7Z |

None of those four records clicked one of its candidate products in the inspected assistant text. That does not select a replacement goal: action text may reflect an error, example, distractor or another trajectory.

Only `webshop_222` / `B01DEDTZ28` matched the earlier inventory product candidates. A scan of 14,248 JSON/JSONL files found **8,316 references**, exactly **2,772 entries in each of three task inventories**. There were no reported matches outside inventory product fields. These are repeated inventory references, not 8,316 model exposures or independent observations. No selected-task candidate match was reported.

The independent provenance review accepted bounded packet consistency but requested changes for full clearance. Raw upstream corpus/catalogue authenticity, historical transformations, stage lineage and actual consumed-input evidence remain outstanding.

## 6. Dependency grouping, contextual review and replay

The catalogue comparison separated candidate goals, human-turn references, assistant text references and assistant click text. All candidate alternatives were retained. Catalogue membership is conditional on the supplied local catalogue, whose historical identity is not authenticated.

Conservative connected components over catalogue-confirmed references produced:

| Quantity | Result |
|---|---:|
| WebShop records covered | 350 |
| Proposed groups | 257 |
| Singleton groups | 218 |
| Multi-record groups | 39 |
| Largest group | 15 records |
| Shared products linking records | 173 |
| Human-turn-only shared links (review reconstruction) | 137 |
| Other-family records not grouped | 1,508 |

Shared demonstrations or distractors can over-group records. Separate components, including singletons, do not establish independence. No stages have been assigned from this graph.

There were 61 catalogue-unmatched associations, spanning 48 records and 27 identifier literals. Independent contextual review classified them as follows:

| Classification | Associations |
|---|---:|
| Numeric literal | 29 |
| Field or column name | 13 |
| Model or brand string | 13 |
| Ordinary word | 2 |
| Other nonproduct token | 2 |
| Possible product identifier | 1 |
| Option label | 1 |

`MULTICOLOR` in `webshop_237` is an option label in the supplied preceding color-option context and proposed click. It is not authenticated click execution. The one possible product identifier is an ISBN-context value, not a verified WebShop ASIN. Context classification does not change catalogue membership or graph edges; every original association remains preserved.

The independent reviewer reconstructed the components and reviewed 105 supplied occurrences. Local intake verified the review content seal, association IDs, classification counts, group IDs and coverage. The subsequent **cluster citation replay passed for 61 associations, 105 occurrences and 257 groups**, with four goal assignments still unresolved. This verifies local text/hash/span consistency, not upstream source authenticity.

## 7. Primary-source searches: negative results and their scope

| Search | Parsed files | Recognized task journals | Matches | Skips/errors/limit overflow |
|---|---:|---:|---:|---|
| Explicit `goal.instruction_text` + `goal.asin` | 14,241 | Not measured in v1 | 0 | 0 / 0 / 0 |
| Expanded goal + AgentBench `output.history` | 14,400 | 0 | 0 goal; 0 trajectory pairs | 0 / 0 / 0 |

The v2 search checks exact adjacent native human/assistant strings against saved observation/response pairs and retains task-input hashes and local indices for potential matches. It does not strip wrappers, normalize case or perform fuzzy matching. Since no journals were recognized, the result is not a failed comparison against a known historical trajectory collection.

Three v2 roots were missing:

- `BackAgentDef/agent-backdoor-attacks/AgentTuning/AgentTuning/AgentBench.old/outputs`
- `BackAgentDef/agent-backdoor-attacks/AgentTuning/AgentTuning/AgentBench.old/src/tasks/webshop/user_session_logs`
- `/dataset/suaq0001/BackAgentDef/results`

Scans are bounded by roots, filename exclusions, formats, schemas and non-followed directory symlinks. Archives, remote sources and unidentified locations were not searched. Repeating the same scans unchanged is unlikely to add evidence. Original collection archives or a source-maintainer mapping are now needed to advance historical lineage authentication.

## 8. Approval ledger

| Review/work product | Accepted scope | Still withheld |
|---|---|---|
| Membership/context | 1,858 whole records; eight quarantined; proposed shared 4,096 cap | Training/resource approval |
| Conversion | Native-format materialization with annotation preservation; CPU checks | Clean provenance and final dataset approval |
| Supervision policy | Mask original false assistant spans; retain context | Original intent authentication |
| Revised adapter | Bounded implementation and CPU evidence; R01–R04 closed | Training launch gate and final attack-corpus masks |
| Provenance review | Supplied packet seals/counts/joins | Full provenance clearance |
| Dependency context | Token classifications and conditional component correctness | Goal identity, execution, stages and split independence |
| Citation replay | Local native citation consistency | Independent historical execution authentication |

No new training, resources, inference, final C/LQ/LO datasets, matched controls or method-matrix execution are authorized. Existing historical approvals are not expanded by these preparation reviews.

## 9. Evidence index for auditors

Cluster paths below are supplied locations, not locally authenticated files. Prefix `P` means `/export/home2/suaq0001/seek_private/three_way_v1/`.

| Evidence | Location / binding |
|---|---|
| Native corpus and masks | `P/materialization-HK4UX8/` |
| Supervision comparison | `P/annotation-mask-comparison-mQJsK6/report.json` |
| Revised adapter receipt | `P/annotation-adapter-revised-MKH8iE/report.json` |
| Gate readiness audit | `P/annotation-gate-audit-2gjBob/report.json` |
| Lineage dossier | `P/lineage-dossier-YuMvl1/report.json` |
| Catalogue comparison | `P/catalogue-lineage-buohse/report.json` |
| Dependency proposal | `P/product-dependency-groups-Vhf0lb/report.json` |
| Context evidence | `P/unmatched-context-OF8BqS/report.json` |
| Citation replay | `P/dependency-review-replay-QFt7iA/report.json` |
| Primary goals v1 | `P/primary-goal-search-Ksi8IY/report.json` |
| Primary lineage v2 | `P/primary-lineage-v2-GvxNBH/report.json` |

Canonical/content identifiers (not interchangeable with raw-file or ZIP hashes):

```text
native_corpus:
8b1165ac799dc4c2bd96e0a13a1c39a098fc73cbb9c8ab453026dc545ae1fed1
annotations:
d4cbd9e0dc138def5eda9d58631ced81e6fc431d5d59a50b34f1aabb576e22f1
supervision_comparison:
5b979d0c7544caacf213a577db9df607efb63541f7b0d10e2a953bd388035b07
supervision_policy_review:
a5bca4dea37329a6cf2f77b6cc36800c70f7c9035c5a57e53408d85858bc5d68
revised_adapter_receipt:
b0f711668cab5938e142cedb92bbc1c803ff98654c77faa7a8a27043c8c68525
revised_adapter_review:
fe176dd4910089fa50c24f7e449733b1cd4da8bf3ba0a79b0fe0dfd395cc79f7
provenance_review:
fbb85e442f038cd1a6f114222f4476c70613ede5e25b5e7fd2ba1d3019677be8
catalogue_comparison:
70f3402c73752c2d0458d215f5b42731bd840c0c5d7966eca0bc421a319b6ddf
dependency_proposal:
e457f57e3fe85a4757bf8211940f28dbdcf744d4a1c479e9e2af8e354e9ab8ee
context_evidence:
4e4c3bb979ae2644f70db7bfbce9335c86584595692bd52534d154b35e96fbaf
dependency_context_review:
293c80bf099b91cef9def3c918f64071354246ea9be2a862f82812b9ec46aad9
citation_replay:
8a43ea088bff08611d6c12622471fae282c73a61b87170299d7161dedfbf3382
primary_goals_v1:
d828f9657f5893fa1a4837f730a3c4f72c969db57b359b5e57e0e2ee4fffc480
primary_lineage_v2:
7a8699b25490ddceaf4a532265de0d990b77e335af3fc7966dc1cf86a5fcf75b
```

Independent review files are archived in [the review directory](../../three_way_experiments_2026_10_19/reviews/). See also [adapter review status](../../three_way_experiments_2026_10_19/trainer_integration/CPU_REVIEW_STATUS.md), [dependency review status](../../three_way_experiments_2026_10_19/DEPENDENCY_CONTEXT_REVIEW_STATUS.md) and [primary-source search scope](../../three_way_experiments_2026_10_19/PRIMARY_GOAL_SEARCH.md).

## 10. Validation scope and remaining audit questions

Recent local targeted suites reported four citation-review tests, three unmatched-context tests, four dependency-group tests and seven primary-search tests passing. These are engineering checks, not independent scientific replications. They were run at their implementation checkpoints, not as a new combined full-suite audit for this report. Cluster adapter/native-tokenizer checks are separately attributed above. No GPU inference, training, downloads, paid APIs or job submissions were performed locally for this preparation work.

The auditor should assess:

1. Whether the preserved source-to-native conversion chain and narrow approvals are correctly bound, without promoting local cached equality to upstream authentication.
2. Whether the substantial supervision change is acceptable for the proposed shared recipe, including its family imbalance, and whether every final corpus will receive its own native mask/target audit.
3. Whether all four ambiguous goal candidate sets and all catalogue-unmatched references remain visible, without replacing goals from click text.
4. Whether component correctness is kept separate from independence, especially shared demonstrations, templates and the 1,508 records outside the WebShop graph.
5. Whether any original task/session archives or publisher mappings can supply missing lineage. If not, the historical-lineage claim must remain unresolved rather than be cleared by repeated negative scans.
6. Whether historical implemented effects, finite constructed development results and semantic interpretation remain separate, with failed controls and failed qualitative reviews retained.

Before new training or readiness execution, the project still needs authenticated source/lineage evidence, reviewed stage assignments and split checks, reviewed query/observation slots and private markers, final C/LQ/LO corpora with separate native mask receipts, an installed/reviewed launch gate, shared initialization/recipe and matched-control evidence, and explicit resource/execution approval. A genuine clean-policy dependency must not be reported as a false backdoor certificate.
