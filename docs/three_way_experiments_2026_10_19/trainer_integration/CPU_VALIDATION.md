# CPU validation acceptance specification

These are required future checks, not a claim that integration exists or passed.

| Check | Required evidence |
|---|---|
| Whole-record binding | Duplicate IDs, stale corpus/sidecar digest, swapped IDs, changed text or reordered turns fail before yielding examples |
| Annotation schema | Missing, null, integer, string or conflicting assistant loss values fail; every new attack example has explicit booleans |
| Label boundaries | Independently expected masks for true/false interleavings, first/last assistant turns, terminators, adjacent special tokens, repeated text and separator spaces |
| Context invariance | Exact baseline/adapter equality of input_ids and attention_mask per example and collated batch; label deltas only in false spans |
| Causal alignment | Labels remain at native token positions; no double shifting; supervised-target positions survive standard causal shift |
| Fail-closed behavior | Empty true targets, truncation, zero supervision, unresolved boundaries and native-mask mismatches produce explicit failure |
| Dataset and collator | Exercise eager/lazy paths actually used by the new entry point, cache replay, shuffled indexing and multi-example batches without loading weights |
| Source binding | Reject changed adapter/trainer/tokenizer/template/cap and stale cached boundaries |
| Receipt gating | Reject old native-only receipts, comparison receipts and unreviewed policies as integrated-training proof |
| Attack-target retention | Explicit target turn/span binding; target labels retained in lexical examples and reviewed benign counterparts; missing targets fail |
| Whole benign corpus | Independent expected-mask comparison for all 1,858 records at 4,096; all counts and before/after token hashes retained |
| Final corpora | New complete real-tokenizer receipts for each final C/LQ/LO corpus, grouped by task family, with no inherited benign-only clearance |

Independent expected-mask fixtures must not call annotation_mask to create their
expected labels. Native model-specific CPU replay requires the pinned local
FastChat and tokenizer assets. Unit tests on synthetic arrays cannot substitute
for that replay. Missing cluster dependencies are reported, not downloaded.

Record command, environment/package versions, source hashes, input hashes,
failures, per-record totals and adapter/collator tensor identities. All receipts
retain model_calls=0 and execution_authorized=false. Training approval is a
separate later decision; passing these checks does not establish provenance.
