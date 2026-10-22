# Annotation-aware trainer integration proposal

Status: proposed implementation, not installed or verified. This proposal requests
review of additive code and CPU validation scope only. It grants no execution,
training, inference, resource, final corpus, provenance or split approval.

## Bindings and intended behavior

Policy: explicit_false_assistant_labels_ignored_v1.
Policy approval: a5bca4dea37329a6cf2f77b6cc36800c70f7c9035c5a57e53408d85858bc5d68.
Comparison: 5b979d0c7544caacf213a577db9df607efb63541f7b0d10e2a953bd388035b07.
Native corpus: 8b1165ac799dc4c2bd96e0a13a1c39a098fc73cbb9c8ab453026dc545ae1fed1.
Annotations: d4cbd9e0dc138def5eda9d58631ced81e6fc431d5d59a50b34f1aabb576e22f1.
Reported cluster trainer SHA-256: 9d175d06bd289fb4a6bec85b680d6e3933491d497a2e99e99707be855e229242.
The trainer bytes are unavailable in the local checkout; implementation must
verify the cluster source and review its dataset/collator interfaces first.

Every conversation remains intact. Input IDs, attention masks, roles, ordering,
text, tokenizer, native template and context cap remain identical to the bound
baseline. False assistant spans, including their native terminators, become -100
labels. True assistant labels stay unchanged. Human, padding and other native
non-target spans remain masked. No default annotations are permitted.

## Proposed integration boundary

Add a separately named Seek dataset/preprocessing adapter, selected explicitly
by a new gated training entry point. Preserve FastChat/train.py and its default
behavior. Do not globally monkey-patch preprocessing, rewrite historical corpora,
or change agent_eval.sh, Shield, Stage I or the authoritative registry.

The adapter obtains the exact native tensors and independently verified native
assistant boundaries using the existing native_record_audit and annotation_mask
components. Cache verified boundaries per full record/content/tokenizer/template/
source hash, never solely by row index. Dataset reordering must preserve binding.
A separate CPU loader test must exercise the actual adapter, collator and batch
path; invoking annotation_mask alone is not integration verification.

Return the same input_ids and attention_mask, with only the declared labels
changed. Keep PyTorch causal-label shifting in the model unchanged; do not shift
labels a second time in the adapter. Verify dtype, shape and padding after collation.
Do not invoke Trainer.train, model.from_pretrained, forward, backward or optimizer
steps during CPU validation. No weight files are needed.

## Annotation and corpus contract

For preserved native files, use the original sidecar. Verify corpus/sidecar seals,
unique record IDs, exact per-record content hash, turn count, ordinal and role.
The current sidecar binds all turn text through the complete record digest.
Explicit true/false booleans are required on every assistant turn. Reject missing,
conflicting, non-boolean, reordered or unmatched annotations; do not infer flags
from text, task family, turn position or the older loss-blind recipe.

Each newly assembled C/LQ/LO example needs an immutable annotation sidecar with
its own record digest and explicit flags. Record targeted action turn ordinals
and hashes. Verify intended attack-action labels are true and benign counterparts
have the reviewed analogous supervision. A shared policy alone does not prove
controls are matched. No final attack corpora are approved here.

## Gates and receipts

Reject truncation, unstable prefix boundaries, native-label discrepancies, absent
positive targets and stale source bindings. Produce fresh immutable adapter
validation receipts; never relabel the existing comparison as integration passed.

Before preparing a training command, require distinct schema/policy/version and
exact adapter, trainer, tokenizer, template, corpus and sidecar bindings. A legacy
native mask receipt must not satisfy an annotation-aware recipe. Maintain the
existing provenance, initialization, split, resource and execution gates.

Revalidate each final assembled C/LQ/LO corpus. Historical supervision counts are
regression expectations only for the exact bound benign corpus and environment:
609,507 before; 434,727 after; 174,780 removed; 4,671 false and 9,675 true turns.
Different final corpora require their own counts, not forced agreement.

## Rollback and limits

The new path is opt-in and can be left unused; no historical artifact needs
rollback. Preserve failed receipts and version changes. Reusing the same masking
policy does not establish training equivalence to legacy SQ/SO or benign utility.
Upstream annotation author intent remains unauthenticated.
