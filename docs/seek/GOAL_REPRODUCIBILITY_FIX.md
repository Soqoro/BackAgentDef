# Recovering from goal-order fingerprint mismatch

Cluster collection jobs 1083069_0 and 1083069_1 loaded their victim checkpoints,
then failed before collection proposals with `environment goal order fingerprint
mismatch`. The dependent replay job 1083071 could not run.

The Seek wrapper had not seeded environment construction. WebShop's
`engine/engine.py:generate_product_prices` calls `random.uniform`, and
`engine/goal.py:get_synthetic_goals` calls `random.sample` to choose price limits.
`SimServer` applies `random.seed(233)` only after constructing those goals. Thus
identical goal counts/product IDs need not mean identical instructions or prices.
The earlier inventory's 186253 goals and 3259 unique products do not resolve this.

Seek now seeds Python random to 42 immediately before environment construction,
then restores the caller's RNG state even on failure. No upstream WebShop or
Stage I source is modified. The existing instruction/goal-object checks remain.
The environment fingerprint now includes the Seek collection wrapper as well as
WebShop source. CPU fake-environment tests reproduce price/limit randomness and
verify matching full goal fingerprints across differing caller RNG states.
Real cluster collection/replay verification is still pending.

## Restart commands (cluster, after syncing the fix)

Do not resume v1 with new source or replace its inventory/task manifest. Preserve
old logs/results and create the fresh v2 configs below. The helper reuses existing
asset/registry references and defender settings without downloading or training.
It updates the environment source hash and clears goal-order identity pending the
new inventory. It refuses an existing output directory or reused run suffix.

```bash
cd ~/BackAgentDef
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 CONDA_NO_PLUGINS=true
conda activate webshop_torchfix
export SEEK_REPO_ROOT="$PWD" SEEK_RUN_ROOT="$PWD/results/seek"
export CONDA_ENV=webshop_torchfix
export SBATCH_PARTITION=PH100q SBATCH_CPUS_PER_TASK=4
export SBATCH_GRES=gpu:1 SBATCH_MEM_PER_NODE=96G SBATCH_TIMELIMIT=02:00:00
unset SEEK_DRY_RUN SEEK_AGENT_MODEL

python docs/seek/restart_pilot.py \
  --source-dir configs/seek/local --output-dir configs/seek/local/v2 --run-suffix v2
python seek_eval.py preflight --config configs/seek/local/v2/inventory.json \
  --phase inventory --metadata-only
bash seek_submit.sh --config configs/seek/local/v2/inventory.json \
  --phase inventory --run-root "$SEEK_RUN_ROOT" --row 0
```

Cancel only the obsolete pending replay job 1083071 if it remains queued. Wait
for the new inventory job to complete with exit 0 before running:

```bash
python seek_eval.py build-manifest \
  --inventory "$SEEK_RUN_ROOT/sneakers_inventory_v2/real/row-0000/inventory.json" \
  --sizes '{"development":4,"discovery":16,"confirmation_removal":8,"confirmation_insertion":8,"reuse":8}' \
  --output configs/seek/local/v2/tasks.json
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'agent-backdoor-attacks/AgentTuning/WebShop')
from seek.storage import immutable_json
p = Path('configs/seek/local/v2')
c = json.loads((p/'pilot_template.json').read_text())
m = json.loads((p/'tasks.json').read_text())
c['environment']['goal_order_hash'] = m['namespace']['goal_order_hash']
immutable_json(p/'cluster_pilot.json', c)
print('New goal-order hash:', c['environment']['goal_order_hash'])
PY
python seek_eval.py preflight --config configs/seek/local/v2/cluster_pilot.json \
  --phase collect --metadata-only
```

Use `configs/seek/local/v2/cluster_pilot.json` for subsequent collection and replay
submissions in the main runbook, and `results/seek/sneakers_pilot_v2` when inspecting
results. Discovery must still wait for valid replay. The Qwen smoke does not need
to be rerun solely for this environment-construction fix. Training provenance and
overlap remain unknown; this repair does not authorize confirmatory claims.
