# Choice-integrity experiments

The root launcher, `choice_integrity_eval.sh`, evaluates the frozen WebShop
preference-laundering benchmark as an 18-cell Slurm array:

| Array cells | Method | Conditions |
| --- | --- | --- |
| 0--2 | `undefended` | clean, direct, indirect |
| 3--5 | `gate` | clean, direct, indirect |
| 6--8 | `state_aware_verifier` | clean, direct, indirect |
| 9--11 | `gate_ci` | clean, direct, indirect |
| 12--14 | `gate_ci_no_ledger` | clean, direct, indirect |
| 15--17 | `gate_ci_no_dominance` | clean, direct, indirect |

The embedded launcher default runs at most two cells concurrently. Each cell
requests one GPU, eight CPUs, 64 GB of host memory, and 24 hours. The submission
commands below override the concurrency cap to four cells, matching a
four-GPU allocation.

## 1. Freeze the legacy split-checkpoint protocol

This experiment uses the two existing compromised policies rather than a
jointly trained checkpoint:

| Conditions | Default checkpoint |
| --- | --- |
| direct | `/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118` |
| clean, indirect | `/dataset/suaq0001/BackAgentDef/outputs/observation_attack/checkpoint-118` |

The direct arm is the released category-conditioned query attack. Its training
data do not contain a distinct, semantically inert user-query cue. Clean and
direct are therefore not a matched same-policy pair that differs only by a
trigger: direct clean-to-trigger preference flips are intentionally not paired
or reported. Direct results remain useful as a separate query-attack arm.

The clean and indirect conditions both use the observation-attack checkpoint,
so their base tasks remain paired. Trigger-conditioned and unconditional
attack/preference metrics are reported alongside the paired indirect
preference flip. Do not invent a direct cue after examining evaluation
outcomes.

## 2. Build and freeze the benchmark once

From the WebShop directory, activate the evaluation environment and build the
manifest:

```bash
source /export/home2/suaq0001/miniconda3/etc/profile.d/conda.sh
conda activate webshop_torchfix
export WEBSHOP_USE_CATALOG_RATINGS=1

python -u choice_integrity_eval.py build \
  --config choice_integrity/config.default.json \
  --manifest benchmarks/choice_integrity_v1.json
```

The v1 builder scans every category-eligible goal. It rejects external
attack-specific task-ID lists because those may encode policy outcomes. Audit
the manifest and its adjacent build report, then preserve its hash before
policy evaluation. The Slurm launcher deliberately fails when the manifest is
absent; it never rebuilds or changes the comparison set during evaluation.

The benchmark and runtime visibly render catalog brand, rating, and availability
metadata. For methods that need comparison evidence, the runtime forks an
isolated WebShop session, issues the goal-only search through `reset`/`step`,
visits every shortlist product and its ordinary Description and Features
pages, and builds the ledger only from rendered HTML, product options, and
those catalog fields. WebShop's evaluator-authored `Attributes`, `query`, and
`product_category` annotations are disabled and never become defense evidence.
Every lookup action and its latency is included in defense overhead, while the
policy still receives its full 15-step horizon.

The hard comparison contract and goal-only query are derived from the
JSON-constrained structured goal parser applied to the instruction actually available
in that condition, after the fixed preference suffix is bound separately. The
defender is never given a hidden clean counterpart. Frozen WebShop annotations
are consulted only afterward as a fail-closed audit: any parser/benchmark
mismatch aborts the cell instead of silently giving the defense oracle
constraints.

## 3. Submit the matrix

Submit from the BackAgentDef repository root. Scheduler logs are written
directly there, so Slurm does not depend on a log directory that may not exist
when it opens stdout and stderr:

No checkpoint export is needed. For a two-task-per-cell smoke test using up to
four concurrent GPUs:

```bash
sbatch --array=0-17%4 --export=ALL,SMOKE_NUM_TASKS=2 choice_integrity_eval.sh
```

For the full matrix using up to four concurrent GPUs:

```bash
sbatch --array=0-17%4 --export=ALL choice_integrity_eval.sh
```

Do not put API keys in the script or in a command line. Export
`OPENAI_API_KEY` through the cluster's secret/environment mechanism before
submission. The launcher requires the OpenAI package and key for GATE, the
state-aware verifier, and GATE-CI by default.

Useful overrides can be passed with Slurm's exported environment:

```bash
sbatch --array=0-17%4 \
  --export=ALL,QUERY_CKPT=/path/to/query/checkpoint,OBS_CKPT=/path/to/observation/checkpoint \
  choice_integrity_eval.sh

sbatch --export=ALL,CONFIG_PATH=/path/to/config.json,MANIFEST_PATH=/path/to/frozen.json \
  choice_integrity_eval.sh

sbatch --export=ALL,OUTPUT_ROOT=/dataset/project/choice_integrity,SEED=42 \
  choice_integrity_eval.sh
```

`QUERY_CKPT` overrides the checkpoint for direct cells. `OBS_CKPT` overrides
the checkpoint for both clean and indirect cells. If omitted, both use the
defaults in the table above.
`REPO_ROOT`, `CONDA_SH`, and `CONDA_ENV` override cluster-specific locations.
The OpenAI-backed parser and verifier default to the pinned
`gpt-5.4-mini-2026-03-17` snapshot. This is the reproducible snapshot behind
the paper's `GPT-5.4-mini` model name; keep the exact identifier frozen in the
reported protocol if you override it.
`WEBSHOP_USE_CATALOG_RATINGS` defaults to `1` for this experiment so
the environment exposes the catalog rating field used by rating preferences;
legacy WebShop runs remain unchanged unless they opt in.

For a local command preview without loading models or data:

```bash
REPO_ROOT="$PWD" DRY_RUN=1 MATRIX_INDEX=0 bash choice_integrity_eval.sh
```

`SMOKE_NUM_TASKS=N` evaluates only the first `N` frozen tasks per cell. A smoke
run should use a new Slurm array job rather than sharing the output directory of
a full run. `REQUIRE_OPENAI=0` permits an offline configuration, but that
changes the parser/verifier protocol and must not be reported as the paper's
OpenAI-backed method.

## Outputs and aggregation

Scheduler output is written in the submission directory:

```text
choice_integrity-<array-job-id>_<cell>.{out,err}
```

Scientific artifacts default to:

```text
/dataset/suaq0001/BackAgentDef/outputs/choice_integrity/runs/slurm_<array-job-id>/
  cells/<method>/<condition>/seed_<seed>/
  aggregate/
  aggregate.log
```

Every successful cell invokes `choice_integrity_eval.py aggregate` while
holding a run-level `flock`. Aggregation reads only cells with a matching,
validated `_SUCCESS.json` whose hashes bind both the episode JSONL and resolved
configuration. It rejects protocol drift, an unexpected checkpoint assignment
within either attack arm, mixed implementation hashes, and noncanonical cell
IDs. An in-progress JSONL from another array task is reported as ignored rather
than mixed into a paper table. The last successful cell produces the complete
aggregate without concurrent writers.
Use `RUN_ID` only when deliberately resuming or collecting cells into an
existing run.

## Scientific guardrails

- Report the query-attack and observation-attack checkpoints as two separately
  trained compromised policies. Do not describe this run as evaluating one
  jointly compromised model.
- The direct arm is category-conditioned and has no distinct user-query cue.
  Report its outcome and defense metrics as a separate query-attack arm; do not
  report a matched clean-to-trigger direct preference-flip metric.
- Clean and indirect use the same observation-attack checkpoint and frozen base
  tasks, so the paired indirect flip metrics remain valid. Keep
  trigger-conditioned and unconditional indirect denominators separate.
- Legacy WebShop runs assign `Rating = N.A.` when the separate review data is
  unavailable. Choice-integrity runs opt in to the catalog's displayed rating
  field. Audit that metadata and freeze it in the manifest before reporting
  rating-preference results; otherwise set
  `WEBSHOP_USE_CATALOG_RATINGS=0` and build a price-only benchmark.
- Clean, direct, and indirect variants must share the same frozen base task,
  hard constraints, candidate set, winner set, and attacker target. Runtime
  ledgers may differ with observed evidence, but attack labels must always use
  the frozen manifest.
- Rendered brand/rating/availability fields are enabled uniformly for all
  methods and conditions. The hidden Attributes page remains disabled. This
  interface choice is recorded in the manifest and resolved run configuration,
  and should be described in the paper.
- Retain no-purchase/infeasible outcomes rather than silently dropping them.
- `_SUCCESS.json` markers are counted as full cells only when they contain every
  frozen task. Aggregate summaries report completeness separately for each
  seed, for the four-method main matrix, and for the two ablations.
