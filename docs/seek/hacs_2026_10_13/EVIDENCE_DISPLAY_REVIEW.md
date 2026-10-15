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
  --plan configs/seek/local/hacs_saved_review_v1/plan.json
python docs/seek/hacs_2026_10_13/review_saved_evidence.py token-check \
  --plan configs/seek/local/hacs_saved_review_v1/plan.json \
  --output configs/seek/local/hacs_saved_review_v1/token_preflight.json
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
appropriate partition/allocation is available; the example uses previously
working PA10080q without pinning a node. It requires one GPU with enough memory
for Qwen 27B and the existing isolated environment and locked local weights.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
python - <<'PY'
import json
from pathlib import Path
p=Path('configs/seek/local/hacs_saved_review_v1')
plan=json.loads((p/'plan.json').read_text())
r=json.loads((p/'token_preflight.json').read_text())
assert r['plan_hash']==plan['hash'] and r['ready'] is True
assert len(r['receipts'])==12 and all(x['within_cap'] for x in r['receipts'])
PY
sbatch <<'SH'
#!/usr/bin/env bash
#SBATCH --job-name=seek-review
#SBATCH --partition=PA10080q
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
  --plan configs/seek/local/hacs_saved_review_v1/plan.json \
  --output results/seek/hacs_saved_review_v1 \
  --approve-development
SH
)
```

After completion:

```bash
python -m json.tool results/seek/hacs_saved_review_v1/result.json
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
