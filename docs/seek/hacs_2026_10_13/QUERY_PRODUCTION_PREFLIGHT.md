# Query-only production attestation and execution gate

The independent v3 plan review is **approve_conditional**, with resources and
execution explicitly withheld. The remaining check is serialization/token/weight
and actual runner enforcement. Nothing here changes that review into approval.
The v3 draft and guard are preserved, including their existing hashes.

`run_scoped_query.py` now provides an offline `attest` operation and a separately
gated production `run`. The attestation binds current runner/helper/Seek sources,
the v3 draft, exact local weight/config inventory, backend versions, tokenizer
identity, template hash, system/history, full serialized prompts, UTF-8 hashes,
and complete token IDs. It checks target absence in the full wrapper and input,
exact request text presence and no truncation. Only `policy_input` is passed to
the renderer; cell metadata remains outside the model input.

The metadata adapter reuses LegacyVictim.render and encode without invoking its
model-loading constructor. It loads the existing local tokenizer only and uses
legacy_reset_text plus the production FastChat template. No model weights are
loaded, no generation occurs, and no downloads are allowed. Hashing checkpoint
shards can take several minutes; that is CPU/disk work, not inference.

## Cluster command now authorized: zero-inference verification

After syncing the code, run on the login host:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
python -u docs/seek/hacs_2026_10_13/run_scoped_query.py attest \
  --draft configs/seek/local/hacs_scoped_continuation_v3/draft.json \
  --output results/seek/audits/hacs_scoped_query_preflight_v1
)
```

Share the summary and provide the reviewer `attestation.json` and
`execution_review_template.json` from that directory, plus the runner and guard
source. The template starts pending and binds both draft and attestation hashes.
The attestation may include private full prompts; keep it within the evaluator
workspace. If files or software change, preserve this attestation and create a
new version for review rather than overwriting it.

## Production enforcement

The runner requires a sealed `seek-query-execution-review-v1` artifact with
independent reviewer identity, decision `approve`, explicit resource and execution
approval, three true checks, and exact draft/attestation bindings. Conditional
plan review alone is rejected. A Slurm allocation is required for run. Before
model loading, it repeats offline verification and requires identical attestation
content. It then constructs QueryOnlyVictim, which loads only the bound query
entry. Loaded-model serialization must match the approved tokenizer-only result.

Four ordered probe requests precede the two first-pair anchor repeats. The
existing response journal, runtime binding, scorer and token checks are reused.
One response is processed at a time, so a malformed/unscorable response stops
further calls. Attempts count even on inference failure. No automatic retries or
resumption are supported. Anchor instability suppresses paired interpretation.
Successful runs require six actual attempts and retain both outcomes of each
pair with left-minus-right differences. No positive-only success rule is used.
Original drafts/results and the authoritative registry are never updated.

After explicit execution approval, the CLI inside an allocated job is:

```text
python docs/seek/hacs_2026_10_13/run_scoped_query.py run \
  --draft configs/seek/local/hacs_scoped_continuation_v3/draft.json \
  --attestation results/seek/audits/hacs_scoped_query_preflight_v1/attestation.json \
  --approval PATH_TO_SEPARATE_APPROVED_EXECUTION_REVIEW.json \
  --output results/seek/hacs_scoped_query_v1
```

This is a CLI reference, not a submission instruction. No sbatch command is
provided before approval. The current conditional review leaves all inference
unauthorized. The method matrix remains locked even after a successful diagnostic.

## Verification limits

Synthetic CPU integration checks exercise six-call success, query-only counts,
metadata-excluding scheduled inputs, no seventh call, failed approval, no resume,
first-response malformed failure, inference exception without retry, serialized
target leakage, truncation and checkpoint inventory mismatch. The production
adapter still requires real cluster preflight; tests do not authenticate cluster
files or model behavior. No local inference, weights download, training, paid API,
Slurm submission or git push is part of this implementation.
