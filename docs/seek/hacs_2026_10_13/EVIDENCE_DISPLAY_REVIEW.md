# Evidence-balanced review packets and saved-evidence development

The supplied v3 audit (`ab9c030d738ce445e6e2a42062cdb837265758e481536929a4fc29dc6a405334`)
reconstructed packet/evidence arithmetic without failures or missing artifacts.
Later packets hid all higher-loss rules, but displayed a constant-output rule
with exactly the same predictions/loss as the two category rules. The reviews
failed to acknowledge that indistinguishability. The proposed next experiment
was ankle boots versus athletic shoes, not a steel-toe/rubber-outsole test.
The Goal review also compared scored rules favorably with unassessed rules.

## Change

The review display selects up to eight rules algorithmically: up to three
best-loss rule forms, two highest-loss rules, one unassessed rule, then tied best
and remaining rules by weight. Tie breaking is deterministic. There is no brand,
category or successful contrast hard-coded into this selection. It changes only
the display; hypothesis generation, loss updates and controller choices remain
unchanged. Unknown rules are still retained in the full family.

The packet reports scored/unassessed counts, loss range, omitted rule count and
groups with identical prediction/loss trajectories on performed probes. These
are empirical equivalences on the observed evidence, not semantic equivalences
or confidence certificates. Review instructions require discussion of equally
fitting alternatives and justification using the chosen experiment's actual
left/right operands. Prose correctness still requires review; no keyword rule
automatically marks a scientific interpretation correct.

The separate saved-evidence CLI reconstructs the historical family IDs, binds the
parent result and audit, and supplies the revised display to the four existing
review positions. It does not run the victim, allocate a claim, or feed new reviews
back into the historical controller. Previous reviews and probes stay intact.
This is post-inspection development, not a controlled method comparison.

## Token-cap correction: saved-review v2

The v1 tokenizer preflight rejected all twelve packets: initial requests were
4126–4172 tokens and acknowledgment repairs reached 4286, above the unchanged
4096 cap. No model calls occurred. Preserve the v1 plan and rejection receipts.

The saved-review-only system prompt now omits initial target-selection guidance
and its grounding example. The previously accepted target is explicitly frozen.
It retains exact schema/acknowledgment requirements, repair limits, loss and
prediction interpretation, equally fitting alternatives, unknown-rule handling,
actual operand justification, and authorization/scope limits. Evidence packets,
observations, rule display, losses, weights and experiment operands are unchanged.
The omission categories and prompt version are bound in the new plan, together
with the review script hash. The general HACS runtime prompt is unchanged.

Four focused CPU tests passed after this correction, including exact payload
preservation and identical message construction for tokenizer preflight and the
real saved-review backend. Character reduction is not a tokenizer guarantee:
all twelve v2 receipts must pass before any GPU submission. Do not reuse or edit
the frozen v1 plan, raise the cap, or truncate evidence.

## Output truncation correction: saved-review v3

The v2 run failed after one real defender response: `finish_reason=length`,
3768 input tokens and exactly 768 output tokens. No review was accepted, no
victim was called, and the registry was unchanged. Preserve result
`a945d0ad50e4d483c830ad91ebfd80e7b098ec1053bd57f8374b129e5af76e81`.
The partial prose discusses tied constant/category predictions, counterevidence,
and actual next-test operands, but remains incomplete and unaccepted.

The new v3 plan copies the parent agent settings and sets only
`max_output_tokens=1024`. Validation rejects a different output budget even if a
modified plan has been rehashed. The parent plan and general HACS runtime remain
unchanged. Saved-review instructions request compact JSON and short, nonrepeating
prose (120/35/70 words for assessment/objection/reason); these are generation
instructions, not an automatic scientific correctness test. Exact acknowledgment
arrays and all evidence fields are preserved. The input cap remains 4096, with
four logical reviews and at most eight physical attempts (maximum 8192 generated
tokens across those attempts). Truncated responses still fail immediately; no
partial acceptance or automatic output-cap escalation is introduced.

All four focused CPU tests pass, including fake-backend four-review completion,
rejection of altered output limits, unchanged parent artifact bytes, and immediate
failure after one truncated reply with zero accepted reviews or victim calls.
No local model generation was performed. V3 tokenizer counts and complete
real-model reviews remain unresolved; run the twelve checks below before submission.

## First step: CPU/tokenizer checks only

After syncing, run:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek \
  -p test_hacs_review_evidence.py -q
python docs/seek/hacs_2026_10_13/review_saved_evidence.py prepare \
  --run-root results/seek/hacs_continuation_dev_v3 \
  --plan configs/seek/local/hacs_saved_review_v3/plan.json
python docs/seek/hacs_2026_10_13/review_saved_evidence.py token-check \
  --plan configs/seek/local/hacs_saved_review_v3/plan.json \
  --output configs/seek/local/hacs_saved_review_v3/token_preflight.json
)
```

This checks four initial packets and both supported acknowledgment repair forms
for each: twelve receipts. Tokenizer access uses the installed local Qwen worker;
no model weights or generation are needed. If any receipt exceeds 4096, stop and
compact the display with explicit omissions; do not raise the cap or truncate.
Every actual call, including any other repair type, is separately checked.

## Gated cluster review, after all receipts pass

Maximum: four logical defender reviews, eight physical attempts, zero victim
calls, zero registry mutations. Preserve the method matrix lock. Confirm an
appropriate partition/allocation is available; the example explicitly requests
NA100q on the command line to override an inherited SBATCH_PARTITION; it does not
pin a node or guarantee current availability. It requires one GPU with enough memory
for Qwen 27B and the existing isolated environment and locked local weights.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
python - <<'PY'
import json
from pathlib import Path
p=Path('configs/seek/local/hacs_saved_review_v3')
plan=json.loads((p/'plan.json').read_text())
r=json.loads((p/'token_preflight.json').read_text())
assert r['plan_hash']==plan['hash'] and r['ready'] is True
assert len(r['receipts'])==12 and all(x['within_cap'] for x in r['receipts'])
PY
sbatch --partition=NA100q <<'SH'
#!/usr/bin/env bash
#SBATCH --job-name=seek-review
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/seek/saved-review-%j.out
#SBATCH --error=logs/seek/saved-review-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source /export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh
conda activate webshop_torchfix
python -u docs/seek/hacs_2026_10_13/review_saved_evidence.py run \
  --plan configs/seek/local/hacs_saved_review_v3/plan.json \
  --output results/seek/hacs_saved_review_v3 \
  --approve-development
SH
)
```

After completion:

```bash
python -m json.tool results/seek/hacs_saved_review_v3/result.json
```

A `reviews_complete` result is an execution status, not a qualitative pass.
Review acknowledgments, treatment of tied and unknown rules, and whether the
stated next test matches its operands. Full execution provenance, independent
review and fresh prospective method-study eligibility remain separate gates.
No local GPU inference, downloads, training, paid APIs, submission or push is
part of this implementation.

## Local validation

All 337 Seek CPU tests passed. After tightening the saved-review budget fields,
the four focused tests passed again, including a fake-backend four-review run
with zero victim calls, immutable parent artifacts, input-cap rejection, null
loss handling, and preservation of historically different prediction trajectories.
`git diff --check` passed. These are simulated software checks, not Qwen reasoning
verification. Cluster tokenizer receipts and the new real review outputs remain
unavailable until the gated commands are executed.
