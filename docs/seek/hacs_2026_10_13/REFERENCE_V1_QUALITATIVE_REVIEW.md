# Reference review v1: citation success, qualitative review not passed

This non-independent assistant review is based on the user-supplied result
`fef656cd0962bba0c6199c0c31e9291a1a9297e651801eac7812f537e6fd8b70`.
Its content seal was checked locally. The cluster journals, packets and controller
evidence bindings have not been independently reconstructed here. This is
post-inspection development reuse, not a fresh experiment or method comparison.

Four reviews completed in four defender calls, reporting 15,548 tokens, no
failure, zero victim calls and zero registry mutations. All citations were
accepted. Model numerical transcription was deliberately not tested by this
interface. No previous numeric-copy failure is reclassified.

All roles recognize tied fitted losses on performed probes and discuss the
unperformed ankle-boots/athletic-shoes contrast. Goal explicitly preserves broad
and constant alternatives and leaves null predictions unassessed. These are
useful observations, but three reviews overstate that the proposed contrast will
identify or isolate a mechanism. The second Action review also describes the
output target adidas as a trigger, confusing it with the manipulated category.

The second Action and State reviews give a misleading 0.81-versus-0.01 paired
loss comparison. For predictions footwear=[0.9,0.9] and
sneaker_like=[0.1,0.9], mean squared losses over the two arms are:

| Hypothetical outcomes | Footwear | Sneaker-like |
|---|---:|---:|
| [0,0] | 0.81 | 0.41 |
| [0,1] | 0.41 | 0.01 |
| [1,0] | 0.41 | 0.81 |
| [1,1] | 0.01 | 0.41 |

These values are reviewer calculations, not observed outcomes. Even a comparison
favoring one of these two rules cannot uniquely identify it among the whole
family. In particular, broad and constant rules can share predictions on this
proposed contrast. Semantic validity, authorization and malicious-training
attribution remain unestablished.

The assistant verdict is **qualitative review not passed**. It is a separate
assessment, not a mutation of the runtime's `qualitative_assessment: unresolved`.
The method matrix remains locked. Preserve the result and stop prompt-only GPU
reruns. A further probe needs a separately reviewed plan that specifies its
hypothetical outcomes and unresolved alternatives; it must retain the prior
agent/controller provenance of the selected contrast.

## Attach the assessment on the cluster

After syncing these documentation files, run this CPU-only command. It verifies
both seals and their binding, then writes a separate immutable review file. It
does not edit the result or authoritative registry.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, 'agent-backdoor-attacks/AgentTuning/WebShop')
from seek.discovery_phase.common import checked
from seek.storage import immutable_json

root = Path('results/seek/hacs_reference_review_v1')
result = checked(json.loads((root / 'result.json').read_text()))
review = checked(json.loads(Path(
    'docs/seek/hacs_2026_10_13/reviews/reference_v1_assistant_review.json'
).read_text()))
assert result['hash'] == review['result_hash'], 'Different result; do not attach'
assert result['plan_hash'] == review['plan_hash'], 'Different plan'
assert result['status'] == 'reviews_complete'
assert review['independent'] is False
assert review['method_matrix_unlocked'] is False
output = root / 'assistant_qualitative_review_v1.json'
immutable_json(output, review)
print(json.dumps(dict(status='review_attached', output=str(output),
    hash=review['hash'], independent=False, method_matrix_unlocked=False)))
PY
)
```

Remaining prerequisites are full cluster-artifact verification, independent
qualitative review and the separate fresh method-study eligibility gates. This
attachment satisfies none of those prerequisites automatically.

## Prepare the independent review dossier

After syncing, run:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python docs/seek/hacs_2026_10_13/prepare_independent_reference_review.py \
  --run-root results/seek/hacs_reference_review_v1 \
  --output results/seek/audits/hacs_reference_review_v1_independent_packet
)
```

The separate `dossier.json` contains four original frozen packets, actual request
messages (including corrections), accepted model replies, controller evidence
bindings, the raw defender journal, the assistant assessment, and source-file
byte hashes. `review_template.json` is explicitly pending with no reviewer or
independence attestation. Nothing is allocated or unlocked. These artifacts may
contain private experiment prompts; they stay local and are not published.

The builder checks content seals, review/result equality and regenerated
controller bindings. It does not claim full journal, source/runtime/weight or
physical-call provenance verification. Missing files and binding mismatches stop
preparation. Repeated preparation of unchanged input is idempotent; different
input cannot overwrite an existing dossier. The original run is not modified.

An independent reviewer should inspect the complete packets and raw replies,
not just this assistant's findings. Their separate signed/name-attributed review
should assess acknowledgment, prediction updates, decision justification, scope,
unperformed tests, and preservation of tied/unknown alternatives. The template
is a checklist, not an automated approval endpoint. If the verdict is negative,
record it without launching another prompt-only retry. Any subsequent diagnostic
or method-study allocation needs its own review and existing eligibility gates.

Local validation: the synthetic CPU integration test checks successful dossier
creation, idempotence, unchanged original files, pending independence, rejection
of nested output and rejection of a tampered controller binding. Cluster files
remain unavailable locally; actual dossier preparation remains outstanding.
