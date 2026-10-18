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

## Execution review requested changes: repaired guard and v4

The supplied review `c0980a4a2994177c4b43e4e776bd9a9a4c999e6fec45723ce04c814d07c3943e`
approved the resource ceiling only and withheld execution. Both blocking defects
were reproduced in code review: runtime extras were not rejected, and the CLI
failure branch referenced sys without importing it. Prior artifacts remain failed
or conditional as originally recorded.

The guard now requires exactly the common.responses runtime fields, with hardware
only for real victims: checkpoint_identity, template_hash, tokenizer, dtype,
backend, generation, system and prefix_messages. Extra runtime-level query_target,
request or brand keys are rejected before the backend is called. This complements
the existing full serialization and target-leakage checks. The runner imports sys
and exposes a tested main(argv) returning 2 for saved failure results; the module
entry point uses sys.exit(main()).

The synthetic integration goes through actual common.responses and the production
runner, capturing backend inputs and asserting no metadata forwarding, six-call
ceiling, zero fallback, failure termination, stable/unstable anchors, and unchanged
failure artifacts on forbidden resume. The malformed-output and unstable-anchor
CLI paths both return 2. The integration test's source hash is now in attestation
sources. These are fake-backend software checks, not real inference results.

Run after syncing, in webshop_torchfix:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
mkdir -p results/seek/audits/hacs_scoped_query_preflight_v2
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek \
  -p test_hacs_reference_review.py -v 2>&1 | tee \
  results/seek/audits/hacs_scoped_query_preflight_v2/cpu_integration_tests.txt
python docs/seek/hacs_2026_10_13/rebind_query_guard.py \
  --original configs/seek/local/hacs_scoped_continuation_v3/draft.json \
  --output configs/seek/local/hacs_scoped_continuation_v4
python -u docs/seek/hacs_2026_10_13/run_scoped_query.py attest \
  --draft configs/seek/local/hacs_scoped_continuation_v4/draft.json \
  --output results/seek/audits/hacs_scoped_query_preflight_v2
)
```

The new v4 freezes the repaired guard, keeps the query checkpoint, all cells,
anchors, direction and six-response cap unchanged, and binds the original v3
hash. It remains unapproved. Re-attestation is mandatory because source bytes
changed; the old approved-in-principle resource review cannot authorize this run.
Send the reviewer v4, the new attestation and execution-review template, CPU test
output and these source files (all execution files below are in the attestation):

- docs/seek/hacs_2026_10_13/run_scoped_query.py
- docs/seek/hacs_2026_10_13/query_only_guard.py
- docs/seek/hacs_2026_10_13/bind_query_continuation.py
- agent-backdoor-attacks/AgentTuning/WebShop/seek/discovery_phase/common.py
- agent-backdoor-attacks/AgentTuning/WebShop/seek/victim.py
- agent-backdoor-attacks/AgentTuning/WebShop/seek/storage.py
- agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_hacs_reference_review.py

The template references the full new hashes, so do not copy the old review or
mark pending checks true automatically. No GPU submission is authorized by these
commands. Real cluster re-attestation and renewed execution approval are pending.
