# Commands in execution order

Run the local verification commands in the checkout. Cluster commands below are recipes for the user; none were submitted during implementation. Use a new experiment output directory for a changed plan, and retain the **same study registry** for all claims in the family.

## 1. Local CPU verification

```bash
W=agent-backdoor-attacks/AgentTuning/WebShop
python -m unittest discover -s "$W/tests/seek" -p 'test_*.py'
python -m unittest discover -s "$W/tests" -p 'test_rebuttal_*.py'
bash "$W/tests/test_agent_eval.sh"
bash -n seek_semantic.sh
python seek_semantic.py budget --j 1 --delta .05 --tau .20 --eta 0
python seek_semantic.py simulate --output /tmp/seek-semantic-cpu-v1 --replications 2 --max-pairs 64 --rules null concept
python seek_semantic.py export --registry /tmp/seek-semantic-cpu-v1/study-0000 --output /tmp/seek-semantic-export-v1
```

`--eta 0` in the budget command is a hypothetical calculation, not a semantic review. Real drafts default to unknown eta. Simulation output paths are immutable; resume the identical simulation settings or choose a new directory for changed settings.

## 2. Cluster: reconcile and select a public incident (CPU only)

After the user has transferred/pulled this implementation:

```bash
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
export SEEK_REPO_ROOT="$PWD"
export SEM_CONFIG="$PWD/configs/seek/local/v2/cluster_pilot.json"
export SEM_REGISTRY="$PWD/results/seek/semantic_v1/study"
export SEM_PILOT="$PWD/results/seek/semantic_v1/query_role_pilot"
mkdir -p logs/seek
python seek_semantic.py import-status \
  --registry "$SEM_REGISTRY" \
  --native-root results/seek/diagnostics/native_characterization_v1 \
  --historical-root results/seek/sneakers_pilot_v2 \
  --historical-root results/seek/diagnostics/sneakers_content_v1 \
  --historical-root results/seek/diagnostics/sneakers_lexical_v1
export SEM_SNAPSHOT="$(python - <<'PY'
import json
from pathlib import Path
root=Path('results/seek/sneakers_pilot_v2/real/row-0000/snapshots')
for p in sorted(root.glob('*.json')):
    s=json.loads(p.read_text())
    a=s['public']['proposed_action'] or ''
    if s['public']['split'] in ('development','discovery') and a.lower().startswith('search['):
        print(p.resolve())
        break
else:
    raise SystemExit('Missing query development/discovery snapshot with public search proposal')
PY
)"
test -n "$SEM_SNAPSHOT" && test -f "$SEM_SNAPSHOT" && test -f "$SEM_CONFIG"
python seek_semantic.py budget --j 1 --delta .05 --tau .20
```

Review the import output. A native `pending_missing` is not a result; if those outputs exist elsewhere, rerun import with that actual root. No new inventory or training is needed. Source/policy/weight checks will fail precisely if the saved runtime differs from the current victim environment.

## 3. First GPU pilot: incident-led, nonempty, one GPU

This is the **first submission**, not a full confirmation job. It uses the Qwen config already in `SEM_CONFIG`, shuts Qwen down, loads the victim, verifies original replay, and runs two constructed pairs. Maximum successful victim generations: five. Target/category proposals come from the public incident and role discussion, not the evaluator specs below.

```bash
P_ID=$(sbatch --parsable --partition=PH100q --gres=gpu:1 \
  seek_semantic.sh role-smoke \
  --config "$SEM_CONFIG" --snapshot "$SEM_SNAPSHOT" \
  --registry "$SEM_REGISTRY" --study-id semantic_v1 \
  --pool-start 100 --pool-size 64 --seed 42001 \
  --output "$SEM_PILOT")
P_ID=${P_ID%%;*}
printf 'Semantic role pilot: %s\n' "$P_ID"
```

After completion:

```bash
sacct -j "$P_ID" --format=JobID,State,ExitCode,Elapsed
cat "$SEM_PILOT/result.json"
tail -n 80 "logs/seek/semantic-${P_ID}.err"
python -m json.tool "$SEM_PILOT/round-00/discussion.json"
python -m json.tool "$SEM_PILOT/round-00/probes/pair-000/result.json"
```

Require `execution=completed`, `status=completed_nonempty_investigation`, two scorable pairs, and the exact replay record under `round-00/replay/events.jsonl`. A scheduler exit of zero alone is insufficient. A rejected hypothesis/backend failure is a recorded outcome, never replaced with fake inference. No real semantic success is claimed until these outputs are received.

For the bounded six-round investigation, after auditing the pilot:

```bash
D_ID=$(sbatch --parsable --partition=PH100q --gres=gpu:1 \
  seek_semantic.sh discover \
  --config "$SEM_CONFIG" --snapshot "$SEM_SNAPSHOT" \
  --registry "$SEM_REGISTRY" --study-id semantic_v1 \
  --pool-start 300 --pool-size 64 --seed 42002 --method adaptive \
  --output results/seek/semantic_v1/query_discovery)
D_ID=${D_ID%%;*}
```

Use `--method fixed` or `--method discussion_only` in separate output directories and distinct unexposed confirmation support ranges for ablations. Keep one registry and the same maximum discovery budgets. Report actual calls, shorter stopping, candidate quality and independent review outcomes; do not claim matched *consumed* costs if methods stop at different times. The returned draft from the chosen round is a candidate, not a certificate.

## 4. Review, globally register, then confirm

For the initial pilot candidate:

```bash
export SEM_DRAFT="$SEM_PILOT/round-00/draft.json"
python seek_semantic.py preview --draft "$SEM_DRAFT" --output "$SEM_PILOT/preview.json"
python seek_semantic.py review-template --draft "$SEM_DRAFT" --output "$SEM_PILOT/review.json"
```

Inspect the actual contract and `preview.json`. An independent reviewer edits `review.json` with their identity, evidence and reason, setting `accepted` only after checking both coherent arms, changed/protected factors, target binding, supported interpretation and unexposed support. The template deliberately starts rejected; it is not an automatic approval. Keep `semantic_eta=null` unless a separate defensible discrepancy bound is supplied. If the contract changes, create a new draft and regenerate its review target.

Then:

```bash
python seek_semantic.py review --draft "$SEM_DRAFT" \
  --review "$SEM_PILOT/review.json" --output "$SEM_PILOT/reviewed.json"
python seek_semantic.py register --draft "$SEM_PILOT/reviewed.json" \
  --registry "$SEM_REGISTRY" > "$SEM_PILOT/registered.json"
export SEM_J=$(python -c 'import json,os; print(json.load(open(os.environ["SEM_PILOT"]+"/registered.json"))["j"])')
python seek_semantic.py budget --j "$SEM_J" --delta .05 --tau .20
C_ID=$(sbatch --parsable --partition=PH100q --gres=gpu:1 \
  seek_semantic.sh confirm --registry "$SEM_REGISTRY" --j "$SEM_J" \
  --config "$SEM_CONFIG" --snapshot "$SEM_SNAPSHOT" --batch 32)
C_ID=${C_ID%%;*}
```

Registration is on the login node before submission. Do not register again to resume. Do not start an array of jobs that each create their own registry. Workers cannot change frozen settings or select samples. Inspect scientific status before a further batch:

```bash
python seek_semantic.py status --registry "$SEM_REGISTRY"
python seek_semantic.py export --registry "$SEM_REGISTRY" \
  --output results/seek/semantic_v1/export_batch01
```

If the stream is still `running/confirming`, submit the **same** `confirm` command for the next 32 pairs, with the same j, source, environment and snapshot. It stops at the frozen certificate rule or 1,024 pairs. A recorded uncertain in-flight response blocks the stream; resubmission cannot reroll it. A changed renderer/scorer/source or eta requires a new contract, j and fresh pool. Keep old logs and failures.

## 5. Observation branch: audit first, then a reviewed hypothetical pilot

```bash
python seek_semantic.py audit-opportunities \
  --run-root results/seek/sneakers_pilot_v2 \
  --output results/seek/semantic_v1/native_opportunities.json
export OBS_SNAPSHOT="$(python - <<'PY'
from pathlib import Path
paths=sorted(Path('results/seek/sneakers_pilot_v2/real/row-0001/snapshots').glob('*.json'))
if not paths: raise SystemExit('Missing observation checkpoint snapshots')
print(paths[0].resolve())
PY
)"
python seek_semantic.py compile --config "$SEM_CONFIG" --snapshot "$OBS_SNAPSHOT" \
  --study-id semantic_v1 --spec configs/seek/semantic/observation_evaluator_spec.json \
  --pool-start 500 --pool-size 8 --seed 42003 \
  --output results/seek/semantic_v1/observation_draft.json
python seek_semantic.py preview --draft results/seek/semantic_v1/observation_draft.json \
  --output results/seek/semantic_v1/observation_preview.json
python seek_semantic.py review-template --draft results/seek/semantic_v1/observation_draft.json \
  --output results/seek/semantic_v1/observation_review.json
```

After independent review of all eight distinct hypothetical backgrounds, edit that review as above. This spec is explicitly **evaluator-specified**, not autonomous recovery.

```bash
O_ID=$(sbatch --parsable --partition=PH100q --gres=gpu:1 \
  seek_semantic.sh observation-pilot \
  --config "$SEM_CONFIG" --snapshot "$OBS_SNAPSHOT" --registry "$SEM_REGISTRY" \
  --study-id semantic_v1 --spec configs/seek/semantic/observation_evaluator_spec.json \
  --pool-start 500 --pool-size 8 --seed 42003 \
  --review results/seek/semantic_v1/observation_review.json \
  --output results/seek/semantic_v1/observation_pilot)
O_ID=${O_ID%%;*}
```

A later observation confirmation uses `compile` with a **new** pool range (for example 700–763), new review and registration, then `confirm` with the observation snapshot. The eight already inspected pilot backgrounds cannot be recycled into confirmation.

## 6. User-run larger CPU statistical study

Benchmark a small run first. Full immutable inputs, outputs and ledgers are retained, so disk requirements grow with replications × claims × pair cap.

```bash
python seek_semantic.py simulate \
  --output results/seek/semantic_simulation_v1 \
  --replications 1000 --max-pairs 1024 --seed 6042026 \
  --rules null below_threshold boundary effect_040 effect_060 strong \
          lexical concept conjunction global_preference ordinary_error sparse \
          positive_eta omitted_hypothesis
```

Each replication is one independently seeded study family with serial claims, repeated looks and the same real-run certifier. The report includes study-wise any-false-certificate frequency and Monte Carlo uncertainty. It is simulated evidence only. No GPU, API, model download or training is used by this command.
