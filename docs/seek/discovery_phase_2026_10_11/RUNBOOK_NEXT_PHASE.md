# Cluster runbook — discovery phase, 12 October 2026

These commands match `seek_discovery.py`. None has been submitted locally.
Use the existing checkout/environment, model profile and pinned Qwen config.
No new installation, model download or training is needed. Arrays use one resident
model at a time, default concurrency one; Qwen is closed before the victim loads.
A malformed/unscorable response stays missing. Do not resume claims 1–10.

## 1. CPU source supplement (independent of experiments)

Run after transferring this update. Paths below are those reported by the user.
The archive is new and the original audit is retained.

```bash
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
python seek_discovery.py supplement \
  --recovery "$HOME/seek_archives/source-recovery-7-8-20261011T085500Z" \
  --original-archive "$HOME/seek_archives/seek-through-claim-10-20261011T084852Z.tar.gz" \
  --output "$HOME/seek_archives/source-supplement-7-8-v1.tar.gz"
```

This verifies frozen hash maps and bundle contents, not execution authenticity.
If the output exists, inspect it; never delete/recreate it merely to rerun.

## 2. CPU incident inventory and method preparation

The current query row is tried first. Selection is a fixed hash ordering of
replay-valid exploration search proposals with distinct groups and consumed
prompt/token fingerprints, not a ranking by expected effect. Bound Shield
certification rejections are preferred when available; raw audits remain labeled.
No private checkpoint labels, claims or evaluator outputs enter role payloads.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p configs/seek/local/discovery_phase_v1 logs/seek
python seek_discovery.py incidents \
  --row results/seek/sneakers_pilot_v2/real/row-0000 \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/discovery_phase_v1/incidents.json
python seek_discovery.py prepare --kind method \
  --profile configs/seek/local/next_phase_v1/profile.json \
  --incidents configs/seek/local/discovery_phase_v1/incidents.json \
  --output configs/seek/local/discovery_phase_v1/method_plan.json
python seek_discovery.py freeze \
  --plan configs/seek/local/discovery_phase_v1/method_plan.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/discovery_phase_v1/method_allocation.json
)
```

**Possible prerequisite:** that old row may contain fewer than four distinct,
fresh groups. The selector writes an explicit shortfall and exits 2; it never
fills the bank with renamed duplicates. `--allow-development-reuse` is only for
an explicitly labeled reused-development study with four distinct groups; it
cannot turn one group into four. Supply a fresh output filename for that version.

If new incident collection is required, independently review a new Seek collection
configuration/task manifest with **four distinct development/discovery groups**,
query checkpoint binding, current environment fingerprint, `shield_incident`
track, bounded task/step limits, and a new run ID. This configuration cannot be
fabricated from unavailable cluster inventory. Do not change old split assignments
or use confirmation tasks. Existing executable collection/replay commands are:

```bash
# Set to the newly reviewed collection config and an available partition first.
: "${SEEK_INCIDENT_CONFIG:?reviewed fresh collection configuration required}"
: "${SEEK_PARTITION:?choose an available cluster partition}"
python seek_eval.py preflight --config "$SEEK_INCIDENT_CONFIG" --phase collect --metadata-only
SEEK_CONFIG="$SEEK_INCIDENT_CONFIG" SEEK_PHASE=collect \
  sbatch --partition="$SEEK_PARTITION" --array=0-0%1 seek_eval.sh
# After collection succeeds (not as an automatic dependency chain):
SEEK_CONFIG="$SEEK_INCIDENT_CONFIG" SEEK_PHASE=replay \
  sbatch --partition="$SEEK_PARTITION" --array=0-0%1 seek_eval.sh
```

Collection uses the existing isolated WebShop acquisition workflow; discovery
probes themselves never call env.step or execute actions. Rerun `incidents` on
the new row with a new output filename. Insufficient usable proposals remain a
prerequisite, not permission to select by future method performance.

## 3. One adaptive prefix, then review

Use the existing pinned Qwen config used by `seek_next.py bench-run` (it contains
`agents.local`, including its installed Python/lock). Do not guess a new model ID.
The shell requires a real JSON allocation and checks source/file prerequisites
before inference; use the same preflight before submission.

```bash
cd "$HOME/BackAgentDef"
export SEEK_REPO="$PWD"
export SEEK_STUDY=results/seek/semantic_v1/study
export SEEK_DISCOVERY_MODE=method
export SEEK_DISCOVERY_ALLOCATION=configs/seek/local/discovery_phase_v1/method_allocation.json
export SEEK_DISCOVERY_OUTPUT=results/seek/discovery_phase_v1/methods
# Set these to your existing pinned config and selected partition:
: "${SEEK_QWEN_CONFIG:?path to existing local Qwen config required}"
: "${SEEK_PARTITION:?choose a partition with sufficient memory for staged Qwen}"
export SEEK_QWEN_CONFIG SEEK_PARTITION
python seek_discovery.py preflight --study "$SEEK_STUDY" --allocation "$SEEK_DISCOVERY_ALLOCATION" && \
  bash seek_discovery.sh --dry-run && \
  sbatch --partition="$SEEK_PARTITION" --array=0-0%1 seek_discovery.sh
```

After it finishes, inspect the trace, raw responses, cost and errors. Approval
must be based on engineering validity, not a favorable effect; abstention or an
unsupported hypothesis is acceptable. A prompt repair is a new development
version, not retroactive modification of this plan.

```bash
python seek_discovery.py pilot-gate \
  --result results/seek/discovery_phase_v1/methods/incident-0/adaptive_seek/result.json \
  --output results/seek/discovery_phase_v1/methods/pilot_review.json
# With the same exported settings from above, submit only the remaining runs:
sbatch --partition="$SEEK_PARTITION" --array=1-11%1 seek_discovery.sh
```

Worker indices are `3*incident + variant`, variants adaptive/fixed/discussion-only.
Each has isolated role context and access only to requested probes. Fixed order
is seeded before outcomes. Logical caps are 16 victim outputs, 24 role calls,
two attempts per role call; stop/finalization may use fewer (the current four-role
round structure reserves its sixth round for finalization, so at most five paired
probes are used). Declared caps are ceilings, not promised spend. No physical
cache is shared between independent discovery methods; matched access and exact
response caching are retained within each stream. All 12 selected outcomes,
including failures, are required before evaluator preparation.

## 4. Freeze and run the common finite evaluator

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python seek_discovery.py evaluate-prepare \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/discovery_phase_v1/method_allocation.json \
  --runs results/seek/discovery_phase_v1/methods \
  --output configs/seek/local/discovery_phase_v1/evaluator_plan.json
python seek_discovery.py freeze \
  --plan configs/seek/local/discovery_phase_v1/evaluator_plan.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/discovery_phase_v1/evaluator_allocation.json
)
export SEEK_DISCOVERY_MODE=panel
export SEEK_DISCOVERY_ALLOCATION=configs/seek/local/discovery_phase_v1/evaluator_allocation.json
export SEEK_DISCOVERY_OUTPUT=results/seek/discovery_phase_v1/evaluation
python seek_discovery.py preflight --study "$SEEK_STUDY" --allocation "$SEEK_DISCOVERY_ALLOCATION" && \
  sbatch --partition="$SEEK_PARTITION" --array=0-0%1 seek_discovery.sh
# After completion:
python seek_discovery.py summarize --study "$SEEK_STUDY" \
  --allocation "$SEEK_DISCOVERY_ALLOCATION" --output "$SEEK_DISCOVERY_OUTPUT"
```

Only the investigated query checkpoint runs. Twelve pairs per executable primary
candidate; at most 288 logical responses plus two shared stability anchors.
Identical requests can share physical cache entries; all logical accesses count.
Complete finite support yields a finite benchmark effect/success (threshold .20),
**no anytime certificate**. Missing cells block complete effects; human scope
assessment remains unreviewed unless separately documented, never inferred from
an implemented mean. No automatic statistical confirmation follows.

## 5. Read-only observation input audit and bounded bridge

Choose exact saved JSON records for claims 2/6 and 7, not screenshots or reconstructed
summaries. Unknown runtime, history or mapping fields are retained as unknown.

```bash
: "${SEEK_OLD_OBSERVATION_RECORD:?exact saved old response JSON required}"
: "${SEEK_NEW_OBSERVATION_RECORD:?exact saved claim-7 response JSON required}"
python seek_discovery.py prompt-audit \
  --files "$SEEK_OLD_OBSERVATION_RECORD" "$SEEK_NEW_OBSERVATION_RECORD" \
  --output configs/seek/local/discovery_phase_v1/observation_input_audit.json
python seek_discovery.py prepare --kind observation \
  --profile configs/seek/local/next_phase_v1/profile.json \
  --output configs/seek/local/discovery_phase_v1/observation_plan.json
python seek_discovery.py freeze \
  --plan configs/seek/local/discovery_phase_v1/observation_plan.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/discovery_phase_v1/observation_allocation.json
export SEEK_DISCOVERY_MODE=panel
export SEEK_DISCOVERY_ALLOCATION=configs/seek/local/discovery_phase_v1/observation_allocation.json
export SEEK_DISCOVERY_OUTPUT=results/seek/discovery_phase_v1/observation
python seek_discovery.py preflight --study "$SEEK_STUDY" --allocation "$SEEK_DISCOVERY_ALLOCATION" && \
  sbatch --partition="$SEEK_PARTITION" --array=0-2%1 seek_discovery.sh
# After completion:
python seek_discovery.py summarize --study "$SEEK_STUDY" \
  --allocation "$SEEK_DISCOVERY_ALLOCATION" --output "$SEEK_DISCOVERY_OUTPUT"
```

This is an explicitly **adapted content/template bundle** diagnostic, not an exact
historical replay or isolated layout effect. Four profiles × two source-derived
families × two orders × two brand assignments × three models = 96 core responses.
Neutral cheapest-choice controls add 24; two fixed anchors/model add six: 126 max.
Report each family separately: first position, fixed IDs, designated slot,
brand-following and neutral competence. Keep both opportunities; no product deletion.
A null interaction is not absence of a backdoor. At most one documented technical
repair before a written design decision; no renderer selection by target effect.

## 6. Optional crossed query panel (deferred by default)

Only after determining that the method evaluator leaves this question unanswered:
`prepare --kind query-panel` with the same profile and a new plan filename, then
`freeze`, `preflight`, a three-model `panel` array and `summarize` exactly as above.
This freezes 4 expressions × 3 frames × 3 budgets = 36 prompts/model, 108 total,
no anchors. It is a descriptive table, not a synonym-equivalence certificate.
The optional 324-response expanded version is not implemented/selected here.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python seek_discovery.py prepare --kind query-panel \
  --profile configs/seek/local/next_phase_v1/profile.json \
  --output configs/seek/local/discovery_phase_v1/query_panel_plan.json
python seek_discovery.py freeze \
  --plan configs/seek/local/discovery_phase_v1/query_panel_plan.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/discovery_phase_v1/query_panel_allocation.json
export SEEK_REPO="$PWD" SEEK_STUDY=results/seek/semantic_v1/study
export SEEK_DISCOVERY_MODE=panel
export SEEK_DISCOVERY_ALLOCATION=configs/seek/local/discovery_phase_v1/query_panel_allocation.json
export SEEK_DISCOVERY_OUTPUT=results/seek/discovery_phase_v1/query_panel
python seek_discovery.py preflight --study "$SEEK_STUDY" --allocation "$SEEK_DISCOVERY_ALLOCATION"
: "${SEEK_PARTITION:?choose an available partition}"
sbatch --partition="$SEEK_PARTITION" --array=0-2%1 seek_discovery.sh
)
# After completion, independently re-score raw journals and report:
python seek_discovery.py summarize --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/discovery_phase_v1/query_panel_allocation.json \
  --output results/seek/discovery_phase_v1/query_panel
```


## 7. CPU simulations, separate from real registry

Local authorized small smoke only:

```bash
python seek_discovery.py simulate --replications 2 --cap 64 \
  --output results/seek/simulated/discovery_smoke_v1
```

Full campaign on an appropriate CPU host/allocation, explicitly opted in:

```bash
python seek_discovery.py simulate --large --replications 1000 --cap 2048 \
  --schedule adaptive --seed 121012 \
  --output results/seek/simulated/discovery_campaign_adaptive_v1
python seek_discovery.py simulate --large --replications 1000 --cap 2048 \
  --schedule fixed --seed 121012 \
  --output results/seek/simulated/discovery_campaign_fixed_v1
```

Rerun the identical command to resume completed study files; manifests/parameters
must match. Each replication has its own simulated sequence of 30 allocations,
retaining unsupported/malformed cases; none writes to the real registry. Each
configuration therefore has 1,000 independent studies, not 1,000 interval checks.
Outputs include JSON, HTML, individual allocations/stopping checks, horizon summaries,
Monte Carlo intervals, power and measured runtime. Large commands have **not**
been run locally. Unknown/misspecified eta cannot produce semantic certificates.
