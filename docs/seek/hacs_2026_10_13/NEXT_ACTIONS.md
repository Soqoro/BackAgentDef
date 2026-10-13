# HACS runbook — first gated cluster pilot

These commands match the implemented CLI. No cluster job was run locally. Existing paths below come from the prior cluster runbooks; their presence/current bindings remain prerequisites. Run from the cluster's `webshop_torchfix` environment after syncing the code. Use fresh output directories; do not overwrite old contracts or results.

The new command family is `python seek_discovery.py hacs-*`. Configuration resource JSONs under `configs/seek/hacs/` document caps enforced by preparation; they do not point to invented checkpoints. Runtime profiles and Qwen paths are read from the user's existing pinned files.

## 1. CPU checks and freeze the prospective library

```bash
cd "$HOME/BackAgentDef"
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek -p 'test_hacs.py' -q
python seek_discovery.py hacs-library \
  --domain configs/seek/hacs/domain_v1.json \
  --output configs/seek/local/hacs_v1/library.json
```

Inspect the library's `coverage`, `omitted` candidates and domain prior. Unknown observation/history rules are expected; this query adapter cannot distinguish them. Library creation performs no model calls or registry writes. A human reviews its meaning before allocation. The domain metadata is development-informed, not autonomously discovered.

## 2. Acquire one new group after that freeze

The original four-case bank and pilot/confirmation sources are excluded. The added `--hacs-pilot` flag permits exactly one group; legacy collection still requires at least four. This is up to two acquisition responses plus one replay response, separate from HACS's 16+2 budget. The legacy inventory is category-filtered; this limits the incident cohort and is not an unbiased survey of ordinary shopping exposure. Selection does not filter for an outcome brand or method success.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python docs/seek/prepare_fresh_query.py \
  --query-config configs/seek/local/v2/cluster_pilot.json \
  --environment-config configs/seek/local/base_reference_v2/cluster_pilot.json \
  --inventory results/seek/base_reference_inventory_v2/real/row-0000/inventory.json \
  --profile configs/seek/local/next_phase_v1/profile.json \
  --study results/seek/semantic_v1/study \
  --history-root results/seek \
  --output configs/seek/local/hacs_incident_pilot_v1 \
  --run-id hacs_incident_pilot_v1 --groups 1 --hacs-pilot
python seek_eval.py preflight \
  --config configs/seek/local/hacs_incident_pilot_v1/cluster_pilot.json \
  --phase collect --metadata-only
)
```

If source/template or inventory checks fail, preserve the failure and prepare a current inventory/environment configuration using the existing pipeline. Do not patch old hashes. After reviewing that preparation:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
export SEEK_CONFIG="$PWD/configs/seek/local/hacs_incident_pilot_v1/cluster_pilot.json"
export SEEK_RUN_ROOT="$PWD/results/seek"
export SEEK_PHASE=collect
sbatch --partition=PA100q --array=0-0%1 seek_eval.sh
)
```

PA100q is a previous 7B victim partition, not a claim of current free capacity. Check `sinfo`/your access before submission. After collection succeeds, run replay separately:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python - <<'PY'
import json
from pathlib import Path
p=Path('results/seek/hacs_incident_pilot_v1/real/row-0000/collect.json')
r=json.loads(p.read_text())
assert r['status']=='complete' and r['simulated'] is False and r['captured']==1, r
print(r)
PY
export SEEK_CONFIG="$PWD/configs/seek/local/hacs_incident_pilot_v1/cluster_pilot.json"
export SEEK_RUN_ROOT="$PWD/results/seek"
export SEEK_PHASE=replay
sbatch --partition=PA100q --array=0-0%1 seek_eval.sh
)
```

A missing `collect.json` means inspect the actual job's `sacct` and matching logs; do not submit replay. Replay must be exact, valid and untruncated. One eligible source might be ambiguous; keep an abstention instead of selecting a more successful incident after seeing HACS outcomes.

## 3. Prepare, review and allocate one pilot

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python seek_discovery.py hacs-prepare \
  --row results/seek/hacs_incident_pilot_v1/real/row-0000 \
  --study results/seek/semantic_v1/study \
  --profile configs/seek/local/next_phase_v1/profile.json \
  --qwen-config configs/seek/local/query_incidents_fresh_v1/cluster_pilot.json \
  --library configs/seek/local/hacs_v1/library.json \
  --mode pilot --output configs/seek/local/hacs_v1/pilot.json
python seek_discovery.py hacs-freeze \
  --plan configs/seek/local/hacs_v1/pilot.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/hacs_v1/pilot_allocation.json
)
```

`hacs-freeze` asks for independent review of the prepared design, actual inputs and resources, including attestation of collection after the library freeze. This is a concrete review gate required by the update prompt; the controller does not infer reviewer independence. It allocates the next index under the authoritative registry lock. No assumed `j=11`, empty shell variable, or registry reset is used.

CPU-only tokenizer preflight uses the pinned Qwen Python and local files, with CUDA hidden. It loads no model weights and downloads nothing:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python seek_discovery.py hacs-token-check \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/hacs_v1/pilot_allocation.json \
  --output configs/seek/local/hacs_v1/token_preflight.json
export SEEK_HACS_ALLOCATION="$PWD/configs/seek/local/hacs_v1/pilot_allocation.json"
export SEEK_HACS_OUTPUT="$PWD/results/seek/hacs_pilot_v1"
export SEEK_HACS_MODE=pilot
bash seek_hacs.sh --dry-run
)
```

The preparatory check covers initial packets. Every later actual packet is checked again before inference. An over-cap result stops; it does not silently raise 4096 to the old 8192/10240 settings.

## 4. Submit only this pilot

The existing runbooks document an A100 80 GB on HPCAIq/node14 for Qwen. Current capacity and access are unverified. Use an accessible **80 GB GPU** for the Qwen 27B review and sequential 7B victim phases. One GPU is requested; models are released between phases and use Slurm-visible cuda:0. No physical GPU ID is assigned. The Qwen configuration retains its validated transport schema before the existing sequential adapter selects visible cuda:0.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
export SEEK_STUDY="$PWD/results/seek/semantic_v1/study"
export SEEK_HACS_ALLOCATION="$PWD/configs/seek/local/hacs_v1/pilot_allocation.json"
export SEEK_HACS_OUTPUT="$PWD/results/seek/hacs_pilot_v1"
export SEEK_HACS_MODE=pilot
sbatch --partition=HPCAIq --array=0-0%1 seek_hacs.sh
)
```

After the job finishes:

```bash
cd "$HOME/BackAgentDef"
python -m json.tool results/seek/hacs_pilot_v1/incident-0/hacs_active/result.json
python -m json.tool results/seek/hacs_pilot_v1/incident-0/hacs_active/review_packet.json
```

Inspect `failure.json` if no result exists. Slurm completion is not scientific success. Review decisions, exact evidence/rows, preflights, losses, duplicate accounting, prefixes and actual budgets. A null finding or unresolved family can be an engineering success. A token/schema failure needs bounded repair under a new version, not an unattended retry loop.

## 5. New independent gate, then four fresh groups

```bash
cd "$HOME/BackAgentDef"
python seek_discovery.py hacs-gate \
  --result results/seek/hacs_pilot_v1/incident-0/hacs_active/result.json \
  --output configs/seek/local/hacs_v1/matrix_gate.json
```

This asks for independent engineering review, all four qualitative rubric decisions and separate matrix resources. `independent=false` cannot unlock it. The historical assistant review and old matrix remain unchanged. Do not proceed until this new gate passes.

After the gate, collect/replay four more fresh groups by repeating section 2 with these changes: omit `--hacs-pilot`, use `--groups 4`, output/run ID `hacs_incidents_matrix_v1`; in the collection assertion require `captured==4`; use the matching config/run path. The existing collector's history exclusions include the one pilot group. Acquisition/replay is up to 8+4 calls, separately accounted.

Then:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python seek_discovery.py hacs-prepare \
  --row results/seek/hacs_incidents_matrix_v1/real/row-0000 \
  --study results/seek/semantic_v1/study \
  --profile configs/seek/local/next_phase_v1/profile.json \
  --qwen-config configs/seek/local/query_incidents_fresh_v1/cluster_pilot.json \
  --library configs/seek/local/hacs_v1/library.json \
  --mode matrix --pilot-gate configs/seek/local/hacs_v1/matrix_gate.json \
  --output configs/seek/local/hacs_v1/matrix.json
python seek_discovery.py hacs-freeze \
  --plan configs/seek/local/hacs_v1/matrix.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/hacs_v1/matrix_allocation.json
python seek_discovery.py hacs-token-check \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/hacs_v1/matrix_allocation.json \
  --output configs/seek/local/hacs_v1/matrix_tokens.json
export SEEK_HACS_ALLOCATION="$PWD/configs/seek/local/hacs_v1/matrix_allocation.json"
export SEEK_HACS_OUTPUT="$PWD/results/seek/hacs_matrix_v1"
export SEEK_HACS_MODE=matrix
bash seek_hacs.sh --dry-run
sbatch --partition=HPCAIq --array=0-11%1 seek_hacs.sh
)
```

The matrix uses a prospectively different prompt/budget pool from the pilot. Exposure checks still fail closed on exact consumed overlaps. No renewed allocation may be used to make old prompts look fresh. New future campaigns require a separately reviewed pool/version, not relabeled source IDs.

## 6. Candidate-specific evaluation, separately gated

Reserve all final primaries in **one** common evaluation plan. This retains every failed or abstaining method in the denominator and permits private physical-cache sharing for identical evaluation prompts. No method receives those outcomes during discovery.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python seek_discovery.py hacs-evaluation-batch-prepare \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/hacs_v1/matrix_allocation.json \
  --runs results/seek/hacs_matrix_v1 \
  --output configs/seek/local/hacs_v1/evaluation.json
python seek_discovery.py hacs-freeze \
  --plan configs/seek/local/hacs_v1/evaluation.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/hacs_v1/evaluation_allocation.json
export SEEK_HACS_ALLOCATION="$PWD/configs/seek/local/hacs_v1/evaluation_allocation.json"
export SEEK_HACS_OUTPUT="$PWD/results/seek/hacs_evaluation_v1"
export SEEK_HACS_MODE=evaluation
bash seek_hacs.sh --dry-run
sbatch --partition=PA100q --array=0-0%1 seek_hacs.sh
)
```

All 12 result files must exist; missing runs are unresolved, not dropped. The reviewed budget is 24 core responses plus two anchors per executable primary (maximum 288 core plus 24 anchors). If no method supplies a stable executable primary, preparation stops and the method results retain the negative outcome. This job uses the 7B victim only. It reports finite-support effects, never a sampling certificate. The single-primary `hacs-evaluation-prepare --result ...` interface remains available for a separately scoped pilot review, but is not used for overlapping common-support matrix comparisons.

No automatic statistical confirmation is launched. If a primary merits a certificate, review a new contract through the existing semantic/next-phase CLI with fresh support and the current registry index. The old claim 10 is finished.

## 7. CPU simulation, not model experiments

Small local smoke:

```bash
python seek_discovery.py hacs-simulate \
  --replications 2 --cap 16 --seed 131013 \
  --output results/seek/simulated/hacs_smoke_v1
```

Separately authorized full CPU discovery campaign (does not use a GPU):

```bash
python seek_discovery.py hacs-simulate \
  --replications 1000 --cap 32 --seed 131013 --large \
  --output results/seek/simulated/hacs_campaign_v1
```

The accompanying certificate check is deliberately a 64-block bounded smoke. For the existing longer-horizon power/FWER campaign, run separately after approving CPU resources:

```bash
python seek_discovery.py simulate \
  --replications 1000 --cap 2048 --seed 131013 --schedule adaptive --large \
  --output results/seek/simulated/hacs_confirmation_campaign_v1
```

Report both Monte Carlo intervals and positive-power/censoring, not just zero false certificates. Neither campaign is marked complete until its output exists. The optional observation bridge remains the existing 126-response plan in `../discovery_phase_2026_10_11/RUNBOOK_NEXT_PHASE.md`; inspect its exact inputs/position mappings and obtain its own gate before submitting. It is not automatically unlocked by a query result.
