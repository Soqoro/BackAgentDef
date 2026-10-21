# Seek/HACS and three-way experiments: progress and review report

**As of:** 21 October 2026, Asia/Singapore  
**Prepared by:** Codex implementation assistant; not an independent auditor  
**Evidence basis:** repository artifacts, user-supplied cluster outputs and independent document-review decisions. The cluster filesystem, original training history and model execution were not independently authenticated here.  
**Purpose:** summarize completed work, observed results, approvals and remaining decisions. This report does not modify the authoritative registry or grant execution approval.

## 1. Current assessment

The latest completed milestone is an exact **1,858-record native-format candidate corpus**, materialized under independent conversion approval, with a **passing native mask preflight at 4,096 tokens**. The reported validation found zero truncations, lost assistant turns, unresolved boundaries or mask mismatches. Original annotations and eight quarantined records were preserved separately.

This is a data-preparation and engineering result. It does **not** establish clean-source provenance, train/evaluation independence, matched clean/poisoned checkpoints, attack readiness, semantic mechanism identification or training authorization.

No new C/LQ/LO training or attack-readiness model execution has been reported in this phase. Historical semantic checkpoints remain unmatched legacy models; AgentLM remains an external reference.

The offline comparison against six cached AgentInstruct Parquet files has now completed: all **1,866 full records match with multiplicity**, with zero records unique to either side. All six Parquet SHA-256 values match their cache blob names. This establishes equality with the local cached record population, not global record order, remote authenticity or split independence. The next pending check is the five previously skipped exposure/inventory files.

## 2. Historical scientific results retained for context

These results are carried forward from the [18 October review report](SEEK_PROGRESS_AND_REVIEW_2026-10-18.md) and supplied cluster outputs. They have not been re-certified in this phase. Blocks and constructed backgrounds are not independently trained attacks.

| Claim | Experiment | Blocks | Mean effect | Reported lower bound | Disposition |
|---|---|---:|---:|---:|---|
| 1 | Query category contrast | 37 | 1.000000 | 0.207108 | Implemented effect certified |
| 2 | Observation product-title label contrast | 130 | 0.684615 | 0.200667 | Implemented effect certified |
| 3 | Bare-Llama query reference | 0 | — | — | Invalid inference; malformed action |
| 4 | Bare-Llama observation reference | 0 | — | — | Invalid inference; malformed action |
| 5 | Query paired effect minus AgentLM paired effect | 224 | 1.000000 | 0.200049 | Between-checkpoint implemented effect certified |
| 6 | Observation paired effect minus AgentLM paired effect | 848 | 0.648585 | 0.204404 | Between-checkpoint implemented effect certified |
| 7 | Observation category × brand-assignment pilot | 12 | 0.000000 | — | Pilot complete; first-position selection dominated |
| 8 | Sneakers versus trainers query pilot | 12 | 0.000000 | — | Pilot complete |
| 9 | Sneakers versus casual loafers query pilot | 12 | 1.000000 | — | Pilot complete |
| 10 | Sneakers versus casual loafers confirmation | 64 | 1.000000 | 0.277102 | Implemented effect certified |

Five streams crossed the implemented threshold of 0.20. Semantic discrepancy `eta` remains unknown; these are not unconditional semantic certificates. Claims 3–4 are failed controls, not evidence of zero effect. Claims 1–6 retain historical source/runtime provenance gaps. Matching frozen source maps were recovered for claims 7–8, which does not itself authenticate original execution.

AgentLM's measured paired effect was zero in the reported between-checkpoint comparisons. It is not a matched clean-training control. Claim 7 selected the first displayed item in every cell for all three models; this did not isolate the proposed observation mechanism. Claim 10 had 40 unique backgrounds and 80 unique prompts, with repeated draws/cache use explicitly distinguished from new independent attacks.

### Incident-led discovery and HACS

- An earlier four-incident × three-method study yielded **12 abstentions**, no candidates and zero victim probes. It did not establish method equivalence or superiority.
- Development probes comparing sneakers/trainers yielded `[1,1]` outcomes and zero differences. Broader and narrower category explanations remained tied.
- Role reviews encountered grounding failures, malformed structured fields, context/output limits, and confusion between observed and hypothetical predictions. Schema/citation success was kept separate from qualitative reasoning quality.
- Independent qualitative review did **not pass** the saved role explanations: alternatives, scope and decision justification were insufficient. That verdict was not overwritten by later engineering success.
- A separately reviewed query-only development diagnostic tested **ankle boots versus athletic shoes** under two constructed $150 templates. Both pairs returned `[0,1]`, giving boots-minus-athletic-shoes **−1**. Two anchors were stable; exactly four query probes plus two anchors were used, with no observation/reference calls.
- Reconstruction reported matching attested sources and six physical calls. This supports descriptive Adidas occurrence on those exact prompts, not unique mechanism identification, purchase success, unauthorized execution, poisoning causation or a method-comparison win.

The prior agent-selected contrast was retained. Controller-generated choices and evaluator-specified prompt revisions were disclosed rather than represented as spontaneous LLM discoveries. The method matrix remains locked.

## 3. New three-way implementation

The implementation is additive and reuses Seek/HACS components. C is the proposed common clean arm, LQ/LO the proposed lexical query/observation arms, and SQ/SO the legacy semantic checkpoints unless separately replicated and verified. “Clean” is currently a planned role, not established provenance.

| Component | Implemented capability | Verification boundary |
|---|---|---|
| Training preparation | Native FastChat compilation, cue states, shared recipe checks, native masking and pair-match checks | No trained C/LQ/LO checkpoints yet |
| Arbitrary-marker HACS | Public span candidates, literal markers, protected edits, hypothesis/coverage integration | Real full study not run |
| Readiness pilots | Frozen plans, runtime/token attestations, guarded calls, anchors and empirical gates | No current readiness outcomes |
| Delivery/Shield integration | Separate raw/consumed cues and proposal/execution/purchase accounting | Deployed simulator parity and full-episode utility remain pending |
| Investigation/evaluation | Active/fixed/discussion components, candidate preservation, finite-support evaluation | No completed new method comparison or held-out evaluation |
| CPU simulations | Arbitrary-span scenarios and existing known-rule calibration machinery | Two-study smoke only, not a powered calibration result |
| Preparation CLI | Context reports/drafts, guarded conversion, provenance inventory | Missing evidence remains explicit; no automatic registry or approval writes |

Existing `agent_eval.sh`, Stage I behavior, historical claims/results and the authoritative registry were preserved. No local GPU inference, training, downloads, paid API calls, `sbatch` or `git push` were performed for this implementation.

Primary references: [implementation](../../three_way_experiments_2026_10_19/IMPLEMENTATION.md), [cluster runbook](../../three_way_experiments_2026_10_19/CLUSTER_RUNBOOK.md), and [input schemas](../../three_way_experiments_2026_10_19/INPUT_SCHEMAS.md). Some implementation notes describe earlier checkpoints in development; the later cluster results below supersede their then-pending mask status, without altering old receipts.

## 4. Corpus discovery and source comparison

The cluster contained FastChat training sources, the base Llama model directory, the historical study directory and the required Python environments. Presence alone did not verify provenance.

| Candidate | Records | Measured relationship |
|---|---:|---|
| `AgentInstruct_clean.json` | 1,866 | Candidate original corpus |
| `AgentInstruct_clean.jsonl` | 1,866 | Identical parsed content to the JSON file |
| `query_attack/agentinstruct_all.json` | 1,866 | Identical parsed content and file bytes to the JSON file |
| `query_attack/webshop.json` | 351 | Conversation-text subset of the candidate corpus |
| Query `clean_star_m50.jsonl` | 1,916 | All 1,866 base conversation texts plus 50 others |
| Observation `clean_star_m50.jsonl` | 1,916 | All 1,866 base conversation texts plus 50 others |

The subset/addition comparisons ignored IDs and loss fields; they do not establish full-record equality for the `clean_star` files or clean provenance of the 50 additions. Those additions were not adopted as the new reviewed benign corpus.

Original 1,866-record corpus:

- 14,498 human and 14,498 assistant turns.
- Assistant annotations: **4,726 `loss:false`**, **9,772 `loss:true`**.
- Human annotations: **14,498 `loss:null`**.
- No duplicate IDs or duplicate conversation texts in the reported audit.
- Canonical corpus hash: `5028f967bd7e7a7da21b32bbf51f8a5f917a9212efc37878885981b1e6903c6c`.

## 5. Native supervision and context diagnostics

### Trainer behavior and checker repairs

The inspected FastChat `train.py` has SHA-256:

`9d175d06bd289fb4a6bec85b680d6e3933491d497a2e99e99707be855e229242`

It uses the **Vicuna template**, reads turn `from` and `value`, and ignores `loss`. Therefore `loss:false` does not suppress assistant supervision in this implementation. This is not proof of original annotation intent or historical trainer identity.

The first eight shortest records were Mind2Web examples. An initial checker rejected a SentencePiece token containing the template's final space and the first characters of `Thought`. A narrow boundary correction allowed those eight records to pass. This was not representative WebShop validation.

All 351 WebShop records then showed slow/fast token-ID differences; the displayed first differences occurred immediately after `</s>`. Actual native preprocessing matched the native slow tokenizer. The checker was changed to verify **exact native prefix boundaries**, with unresolved boundaries remaining blocking errors. It did not replace the training tokenizer or relax serialization equality.

### Original corpus at 2,048 tokens

| Population | Aligned | Unresolved | Over cap among aligned | Fully lost assistant turns | Partially truncated turns | Mask mismatch records |
|---|---:|---:|---:|---:|---:|---:|
| WebShop | 351 | 0 | 52 | 80 | 5 | 0 |
| Full candidate corpus | 1,865 | 1 | 374 | 1,322 | 116 | 0 |

The unresolved record, `mind2web_60`, contains an empty assistant response marked `loss:true`. It was classified explicitly rather than repaired. Zero label mismatches applies to aligned, retained tokens, not complete supervision of truncated conversations.

### Context-policy comparison

| Hypothetical cap | Whole records fitting / 1,866 | WebShop fitting / 351 | Suffix examples losing history | Nonempty targets still unfit |
|---|---:|---:|---:|---:|
| 2,048 | 1,492 | 299 | 1,430 | 8 |
| 4,096 | 1,859 | 350 | 43 | 1 |
| 8,192 | 1,864 | 351 | 7 | 0 |

Fit counts include the empty-response record. Family labels were derived from IDs, not independently verified lineage. The model configuration declares 4,096 positions with no configured RoPE scaling; 8,192 was a hypothetical length calculation, not approved model support.

At 2,048, excluding overlong records would remove 308/324 KG conversations. Overlapping suffix examples also change weighting: at 4,096, WebShop would have 4,786 assistant supervision occurrences for 1,642 nonempty assistant turns. This is a turn-occurrence comparison, not a token-weighted gradient estimate. Suffixes can drop the original task goal, and the current trainer does not implement target-only supervision.

## 6. Independently reviewed membership and conversion

### Approved membership/context policy

The independent document review approved 1,858 original whole conversations and eight quarantined records, with a shared proposed 4,096 cap for C/LQ/LO. No overlapping windows, truncation, history rewriting or empty-response repair was approved.

| ID-prefix family | Retained | Quarantined |
|---|---:|---:|
| ALFWorld | 336 | 0 |
| DB | 538 | 0 |
| KG | 318 | 6 |
| Mind2Web | 121 | 1 |
| OS | 195 | 0 |
| WebShop | 350 | 1 |
| **Total** | **1,858** | **8** |

Quarantined: `webshop_242` (4,286 tokens); `kg_69` (5,500), `kg_159` (6,176), `kg_207` (9,131), `kg_213` (8,684), `kg_227` (4,551), `kg_319` (5,740); and empty-response `mind2web_60`. These are policy exclusions, not findings of poisoning.

The retained maximum reported length was 3,949 tokens. The policy accommodates 367 retained nonempty conversations that exceeded 2,048. The 4,096 cap remains a change requiring exact future recipe binding and resource verification before training.

### Approved schema conversion

A separate independent review authorized only materialization and CPU validation. Conversion removes turn-level `loss`, preserves IDs, roles, text and order, and stores original annotations in the audit manifest. It does not approve the eventual training supervision policy or establish author intent for `loss:false`.

A serialization defect in an earlier context report was fixed before these reviews: integer exposure-map keys changed to strings on JSON reload, invalidating sorted-key hashes. Reports were regenerated in new directories with string keys before sealing. Old artifacts were not relabeled or overwritten; round-trip regression tests were added.

### Completed materialization and exact native masks

| Check | Supplied cluster result |
|---|---|
| Native records materialized | 1,858 |
| Native corpus hash | `8b1165ac799dc4c2bd96e0a13a1c39a098fc73cbb9c8ab453026dc545ae1fed1` |
| Original annotations preserved | 4,671 GPT false; 9,675 GPT true; 14,346 human null |
| Native mask status | Passed at 4,096 |
| Aligned records | 1,858 |
| Truncated records / fully lost turns / partial turns | 0 / 0 / 0 |
| Unresolved boundaries / mask mismatches | 0 / 0 |
| Model calls | 0 |
| Training/model-execution authorization | False |

The immutable materialization manifest still says `materialized_pending_native_masks`, because it preceded the mask check. The separate passing receipt supplies subsequent evidence; the manifest was not rewritten.

## 7. Provenance and split investigation: current blocker

A bounded read-only inventory verified materialization/mask bindings and examined selected saved JSON fields for exact equality with training human-turn text:

- 68 candidate files; 63 parsed; five skipped for exceeding 2 MB.
- No file-count limit reached and no scan errors.
- Zero exact text matches in the inspected fields.

This does not establish independence. It excludes token-stream comparisons, JSONL journals and other formats, and equality is not substring/semantic matching. A stored match would not itself prove model execution.

The five skipped files include three roughly 69 MB task inventories, a 2.25 MB task manifest and a 2.18 MB exposure manifest. They remain pending inspection; the absence-of-match result is incomplete.

Tracked READMEs identify AgentInstruct as a possible upstream source. No data-directory provenance sidecars were found. The cache contains six family Parquet files under:

`~/.cache/huggingface/hub/datasets--zai-org--AgentInstruct/snapshots/e252cf78ced8a0ea5f62cfd591784cdbbddbac8a/data/`

Both THUDM and zai-org Arrow cache directories were also found. Cache names and a snapshot directory are local provenance clues, not remote authentication. The initial Parquet comparison could not run in `webshop_torchfix` because `pyarrow` was absent. The comparison then ran in `backdoor-def` and reported 1,866/1,866 exact full-record matches with multiplicity, zero local-only or cache-only records, and matching SHA-256 blob names for all six Parquet files. Thus IDs, conversation text, roles and loss annotations match the cached record population. This comparison does not verify global record order, authenticate the remote revision, or establish clean provenance. The receipt is `~/seek_private/three_way_v1/cached-source-comparison-20261021T072710597576Z.json`. No downloads or model calls were reported.

## 8. Approval ledger and evidence locations

All paths below are cluster paths or repository-relative references; this report does not assert independent access to the cluster files.

| Artifact | Identifier/location |
|---|---|
| Context-policy draft | `0f3603117a404e7aae874c4801df0f4b372cb0612c5fdd94f227ccae5f659797` |
| Context report | `8ff287a7e462962186fc78ba79543d0b4508cdcabcffe43a8a89cc600ad95e9c` |
| Policy review | `23c3d146979867881563819135671d7d92793ae45ed1a5b50b5672aa44849639` |
| Conversion draft | `f7e785b1ab65009dde6227e846cc953750f9b05cd400af137253fa60a46e837d` |
| Conversion review | `ba022d7beb97a96b0f8bb94407019f00471bc5d31ee8bb196bac28d0e4c79c36` |
| Policy/report directory | `~/seek_private/three_way_v1/context-reviewed-draft-R27GWl/` |
| Conversion draft directory | `~/seek_private/three_way_v1/conversion-draft-a5HrYN/` |
| Materialization and passing mask receipt | `~/seek_private/three_way_v1/materialization-HK4UX8/` |
| Provenance inventory | `~/seek_private/three_way_v1/provenance-inventory-0FyVAO/inventory.json` |

The review JSON files are preserved in [the repository review directory](../../three_way_experiments_2026_10_19/reviews/). Independent reviewers disclosed document/source review without original raw corpus access or cluster authentication. Their approvals are narrow: membership/context and conversion/CPU validation. They do not approve training, resources, model inference or the method matrix.

## 9. Tests and what they establish

The [19 October test receipt](../../three_way_experiments_2026_10_19/validation/TEST_RECEIPT.json) records 368 Seek tests (including 25 new tests), 52 Stage I regression tests, 36 `agent_eval` dry-run rows, shell syntax checks and 24 unchanged protected-file hashes. Those were the implementation checkpoint's results, not a fresh full-suite run after every later edit.

Later targeted validation reached **38 passing CPU tests**: 28 three-way, eight context/conversion and two provenance-inventory tests. These cover preparation logic and integrity/failure handling; fake tokenizer/backend tests do not replace the cluster's real native mask check. These counts overlap earlier suites and must not be added as independent validations.

The initial two-study simulation was explicitly smoke-only. Its small sample cannot support calibration, power or method-superiority conclusions. No new powered simulation result is reported here.

## 10. Next work, in order

1. Preserve and review the completed cached-Parquet comparison and its exact file/revision bindings. Resolve upstream authenticity and transformation provenance; do not infer remote authenticity solely from local equality.
2. Inspect the five skipped exposure files and appropriate journals with explicit resource limits. Distinguish inventories, actual consumed prompts/tokens and model-call evidence.
3. Reconstruct original source/transform lineage and task, product, trajectory and template-family identifiers. Do not invent mappings from sequential IDs. Missing lineage is a prerequisite, not an automatically passing split.
4. Review clean provenance, training supervision intent and stage assignments. Reuse existing cross-stage split checks when the required evidence exists.
5. Prepare reviewed, task-grounded query/observation source slots with valid continuations. The native benign corpus is not yet the complete common C/LQ/LO training input-exposure design.
6. Bind exact shared C/LQ/LO recipes, compiler outputs, tokenizer/mask receipts and initialization evidence. Validate resource feasibility; obtain separate training authorization before submission.
7. Run bounded attack-readiness pilots only after training/artifact and pilot reviews. Full confirmation, held-out evaluation and method comparisons remain gated.

**Review conclusion:** the conversion and native-supervision engineering milestone is complete on the supplied cluster evidence. Scientific provenance and split independence are not complete. Historical implemented-effect results remain useful within their frozen scopes; the new matched-control experiments have not yet been trained or executed.
