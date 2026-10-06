# Native-task characterization and observation opportunity audit

The completed lexical pilot reports 24 successful real-model calls and exact
original replays. The query checkpoint adds Adidas to all four requests under
`sneakers`, `sneaker shoes`, and `trainers`; the observation checkpoint adds it to
none. Alias prompts contain zero sneaker/sneakers words. This establishes that
those literal words are not necessary for the observed query behavior in this
one development dependence group. It does not identify semantic activation,
unconditional preference, poisoning causality, or the observation attack's
product-selection mechanism.

## New scope

`native-task-characterization-v1` selects native, unmodified WebShop instructions
from three lexical cohorts: sneaker, shirt and watch requests. Both the request
and its catalogue goal metadata must contain the corresponding cohort word;
ambiguous cross-cohort requests and requests explicitly naming Adidas are
excluded. This is a reproducible lexical cohort definition, not a verified
product taxonomy. Controls are different tasks, never claimed to be valid edits
of the sneaker request.

Preparation constructs the existing local WebShop environment on CPU with seed
42 and **no legacy sneaker filter**, then builds connected components across the
entire native goal universe using shared product ID and identical instruction.
It excludes every component touching an instruction or trajectory in the old
registered task manifest, including old confirmation/reuse groups. Cross-cohort
components are excluded. It selects one deterministic representative from four
remaining groups per cohort, before inspecting any policy outputs. Insufficient
groups abort; the implementation does not fill the sample with sibling variants.
Training overlap remains unknown. Shared language/templates may create residual
dependence beyond these group keys; group membership is not proof of independence.

The preparation output freezes 12 selected goals, their native initial-page
observations and action sets, full native-goal order hash, catalogue/environment
identity, prior-manifest hash, generation settings and checkpoint weight
identities. It never rewrites a category or goal. Old filtered local IDs are not
reused in the new namespace. Each GPU row uses those same frozen initial pages,
loads its existing checkpoint once and makes one proposal plus a no-edit replay
per task: at most 24 calls per checkpoint, 48 total. There is no WebShop stepping,
Qwen call, task rewriting, poisoning, new training or downloading in this phase.
No purchase/reward/attack-success measure is produced from initial search probes.

The two existing checkpoints are compared descriptively. Neither is an independent
clean model, and `CLEAN_CKPT` is not used. An independently documented clean
checkpoint remains a prerequisite for attributing differences to poisoned
training. No training corpus is required to run this descriptive pilot.

`audit-opportunities` separately reads **all** existing real snapshot rows. It
reports legal product IDs, available title bindings, Adidas-title and other-title
options, missing bindings, original instructions, page evidence and recorded
proposal measurements. Page inclusion never depends on which product was chosen.
Suitability remains **unverified**, because title/price visibility does not prove
closure, sole, size, availability or all other requirements are satisfied. No
page is automatically declared eligible for a product-text intervention. Product
facts remain protected. If the old pages have no Adidas-titled options, they do
not provide the desired brand-choice opportunity; searching explicitly for Adidas
to create one would introduce another experimental condition, not fix that absence.

## Commands, in order

Sync the new files to the cluster first. Use the existing `webshop_torchfix`
environment and original v2 config. These commands create new artifacts.

1. Run the lightweight CPU opportunity audit:

```bash
cd ~/BackAgentDef
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python docs/seek/characterize.py audit-opportunities \
  --run-root results/seek/sneakers_pilot_v2 \
  --output results/seek/diagnostics/observation_opportunities_v1.json
```

This command imports no model or WebShop runtime, makes zero model calls, and
writes a new report without changing snapshots.

2. Prepare the broader native task plan on CPU:

```bash
export SEEK_CHARACTERIZE_ROOT="$PWD/results/seek/diagnostics/native_characterization_v1"
python docs/seek/characterize.py prepare \
  --config configs/seek/local/v2/cluster_pilot.json \
  --per-cohort 4 \
  --output "$SEEK_CHARACTERIZE_ROOT"
```

Preparation loads the full native goal universe and may take substantial CPU time
and RAM, as the previous environment inventory did. It uses local spaCy/WebShop/
Lucene assets from the working cluster environment; it does not load a victim or
initialize CUDA, even though WebShop imports torch. No dependencies or assets are
automatically installed. Asset-hash, missing-package or insufficient-group errors
must be resolved before submission; do not substitute unrelated numeric task IDs.
If your site requires heavy CPU preparation inside an allocation, run this same
command in a permitted CPU allocation using the site's documented resource policy.
No CPU partition name or current scheduler availability is assumed here.

Inspect the printed selection counts and `plan.json`. Expected selection is four
unused groups each for sneaker, shirt and watch, with 24 maximum calls per row.
No synonym review is needed because these are native unmodified tasks. This
preparation has not been run on real cluster assets locally; eligibility and
available counts are still prerequisites, not locally verified facts.

3. Dry-run both victim workers:

```bash
SEEK_DRY_RUN=1 SEEK_ROW=0 bash seek_characterize.sh
SEEK_DRY_RUN=1 SEEK_ROW=1 bash seek_characterize.sh
```

Both should report `ready`, 12 tasks and 24 maximum victim calls. Dry-run does not
activate conda, load weights, call models or query the environment.

4. After preparation and both checks pass, submit the bounded pilot yourself:

```bash
mkdir -p logs/seek
export CONDA_ENV=webshop_torchfix
B_ID=$(sbatch --parsable --array=0-1%1 --partition=PH100q seek_characterize.sh)
B_ID=${B_ID%%;*}
printf 'Characterization job: %s\n' "$B_ID"
```

The new script requests one visible GPU, four CPUs, 64 GB host memory and 30
minutes per row, concurrency one, using the supplied PH100q partition. Slurm
controls availability; no node is pinned or physical CUDA index reassigned.
No job was submitted locally. Existing outputs are refused, including interrupted
ones; retain their logs and choose a new output for a rerun.

5. Inspect and return reports:

```bash
sacct -j "$B_ID" --format=JobID,State,ExitCode,Elapsed
for ROW in 0000 0001; do
  python -m json.tool "$SEEK_CHARACTERIZE_ROOT/row-$ROW/result.json"
done
```

Return both `result.json` files and `observation_opportunities_v1.json` (the full
report, including page evidence). Keep all manifests, pre-call snapshots and events
on the cluster. The cohort summaries retain selected, replay-valid and scorable
denominators; unavailable outcomes are null, not zero. Original-versus-replay raw
answer equality is recorded separately from parsed-action equality.

## Interpretation and remaining prerequisites

Adidas appearing across all three cohorts would motivate testing a broad brand
preference; a sneaker-concentrated pattern would motivate a category association.
Neither comparison by itself proves a backdoor: these are different task
populations and no matched clean policy is supplied. Four groups per cohort is a
bounded pilot, not a confirmatory sample or a recovery rate.

A substantive observation-model selection experiment still requires pages with
both brand options, independently reviewed suitability evidence, and an
intervention that preserves all legitimate facts. The opportunity audit makes
those requirements concrete instead of silently renaming products or counting
unavailable Adidas choices as attack failures. Such evidence may require a new,
separately designed collection; it is not fabricated from current snapshots.

CPU tests exercise grouping/exclusion through connected products/instructions,
metadata-backed cohort assignment, insufficient-group errors, a stub native
preparation adapter, full fake capture/replay, opportunity/suitability separation,
missingness, model-free imports and Slurm dry-run quoting. These are simulated
protocol checks, not real-asset, real-model or GPU verification.
