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

At most two cells run concurrently. Each cell requests one GPU, eight CPUs,
64 GB of host memory, and 24 hours.

## 1. Fix and freeze the attack protocol

Set `benchmark.direct_trigger_text` in `choice_integrity/config.default.json` to
the exact user-query cue used when the compromised model was trained. The
default is deliberately empty and benchmark construction rejects it: an empty
cue would make clean and direct inputs identical under greedy decoding.

The legacy poisoning artifacts teach category-conditioned target-brand behavior
but do not put a distinct cue in the policy's query-channel input. Consequently,
they cannot support the paper's matched clean/direct claim. Choose the cue
before evaluation, regenerate the direct traces with the updated generator, and
train one checkpoint on the clean data plus 50 cue-bearing direct and 50
indirect traces:

```bash
python create.py \
  --attack_type both \
  --direct_trigger_text 'YOUR_FROZEN_CUE' \
  --num_samples 100 \
  --output_dir ./generated_choice_integrity
```

Apply the paper's reward filter and sampling rule, then jointly fine-tune the
single compromised policy. `create.py` now stores the cue in the serialized
initial instruction and attack metadata and refuses to save a direct trace when
the cue is missing. Use the same exact cue in the benchmark config; do not
invent or tune it after looking at evaluation outcomes.

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

The paired protocol requires one compromised checkpoint containing both direct
and indirect behavior. Add
`choice_integrity_provenance.json` to that checkpoint:

```json
{
  "schema_version": 1,
  "training_data_sha256": {
    "clean": "<64 lowercase hex characters>",
    "direct": "<64 lowercase hex characters>",
    "indirect": "<64 lowercase hex characters>"
  },
  "poisoned_trajectory_counts": {
    "direct": 50,
    "indirect": 50
  },
  "direct_trigger_text": "YOUR_FROZEN_CUE"
}
```

Compute the dataset hashes with `sha256sum` after the final filtering/sampling
step. The evaluator checks this declaration against the manifest and
content-hashes the complete checkpoint once per Slurm run. Export the
checkpoint once, then submit:

```bash
export MODEL_CHECKPOINT=/path/to/combined/compromised/checkpoint
sbatch choice_integrity_eval.sh
```

Do not put API keys in the script or in a command line. Export
`OPENAI_API_KEY` through the cluster's secret/environment mechanism before
submission. The launcher requires the OpenAI package and key for GATE, the
state-aware verifier, and GATE-CI by default.

Useful overrides can be passed with Slurm's exported environment:

```bash
sbatch --export=ALL,SMOKE_NUM_TASKS=2 choice_integrity_eval.sh

sbatch --export=ALL,CONFIG_PATH=/path/to/config.json,MANIFEST_PATH=/path/to/frozen.json \
  choice_integrity_eval.sh

sbatch --export=ALL,OUTPUT_ROOT=/dataset/project/choice_integrity,SEED=42 \
  choice_integrity_eval.sh
```

`MODEL_CHECKPOINT` is intentionally shared by every method and condition.
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
configuration. It rejects protocol drift, mixed implementation/checkpoint
content hashes or training provenance, and noncanonical cell IDs. An in-progress
JSONL from another array task is reported as ignored rather than mixed into a
paper table. The last successful cell produces the complete aggregate without
concurrent writers.
Use `RUN_ID` only when deliberately resuming or collecting cells into an
existing run.

## Scientific guardrails

- The repository currently has separate query-attack and observation-attack
  checkpoints, while the manuscript describes one combined compromised model.
  The launcher has no split-checkpoint fallback and rejects checkpoints without
  the joint-training provenance file.
- The direct cue is not specified in the current manuscript or released
  poisoning data. The updated generator supports a real cue, but the cue must be
  selected and the combined model retrained before this experiment can support
  a paired-trigger claim.
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
  interface choice is recorded in the manifest/provenance and should be
  described in the paper.
- Keep trigger-conditioned and unconditional indirect denominators separate,
  and retain no-purchase/infeasible outcomes rather than silently dropping
  them.
- `_SUCCESS.json` markers are counted as full cells only when they contain every
  frozen task. Aggregate summaries report completeness separately for each
  seed, for the four-method main matrix, and for the two ablations.
