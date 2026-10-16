# Reference-based saved-evidence review v1

This is a new development interface, not a repaired v7 result. V7 failed after
its fourth review repeatedly copied a proposed prediction into an observed
prediction field. Keep that result and its journals intact.

The model now returns `evidence_refs` with ordered `observed_ids` and a
`proposed_id`, alongside its selected hypotheses, decision, and prose. It does
not return `observed` or `numeric_evidence` arrays. The controller validates the
citations and writes sealed `evidence_bindings/NN.json` containing the arithmetic,
packet hash and model-review hash. Bindings explicitly identify controller
ownership. A temporary controller-supplied projection is used only to reuse the
existing base schema validator; it is never saved as model-authored observation.

Citation success does not demonstrate numerical understanding. Prose can still
confuse observed and proposed evidence, overlook tied alternatives, or overclaim.
Every result therefore leaves `qualitative_assessment` unresolved and the method
matrix locked. This interface cannot be pooled with numeric transcription success
rates. No new victim probes, registry mutations, or scientific confirmation occur.

The builder preserves the prior agent-selected experiments. It requires the
completed v5 review as a source of four choices and the intact continuation v3
artifacts. It does not replace those choices with an evaluator-selected contrast.
Old numeric validation and CLI remain available unchanged. Plans bind both the
new script and the legacy packet-builder bytes, plus current execution sources.

## CPU checks and cluster metadata preparation

After syncing the code, run on the cluster login host:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek \
  -p test_hacs_reference_review.py -v

python docs/seek/hacs_2026_10_13/review_evidence_refs.py prepare \
  --run-root results/seek/hacs_continuation_dev_v3 \
  --prior-review results/seek/hacs_saved_review_v5/result.json \
  --plan configs/seek/local/hacs_reference_review_v1/plan.json

python docs/seek/hacs_2026_10_13/review_evidence_refs.py token-check \
  --plan configs/seek/local/hacs_reference_review_v1/plan.json \
  --output configs/seek/local/hacs_reference_review_v1/token_preflight.json
)
```

These commands do not generate model responses. They require the existing locked
Qwen tokenizer and isolated metadata worker. Preflight covers four initial
packets and representative citation repairs. Each actual later packet is checked
again. A failed cap check must not be bypassed or solved by silently raising the
cap. Input cap remains 4096; output budget remains 1024; maximum is four logical
reviews/eight physical attempts and zero victim calls.

Review these new receipts and the changed interface before submitting any GPU
job. The exact runner, for an approved allocated job, is:

```bash
python docs/seek/hacs_2026_10_13/review_evidence_refs.py run \
  --plan configs/seek/local/hacs_reference_review_v1/plan.json \
  --output results/seek/hacs_reference_review_v1 \
  --approve-development
```

The runner requires Slurm, repeats preflight, refuses existing started output,
and retains bounded failures. No submission is included here: current GPU
availability and the new cluster token receipts are unresolved prerequisites.
After execution, review all accepted prose independently against both performed
and proposed tables; neither citation validation nor controller bindings unlock
the method matrix.

## Local verification scope

CPU tests use synthetic roles and victims. They check parent artifact preservation,
controller ownership, citation/phase/duplicate rejection, scientific-choice
preservation on repair, failure after bounded retries, and zero victim calls in
the saved review runner. These tests do not establish real Qwen reasoning quality
or cluster readiness.
