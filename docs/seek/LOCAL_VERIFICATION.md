# Local verification record

Verified on 2026-09-26. All local runtime evidence is CPU-only. Seek victim and
defender outputs below are **simulated**. No real victim model, GPU, paid API,
model download, training or `sbatch` submission was run.

The original baseline checks were run before implementation and again afterward:

```bash
W=agent-backdoor-attacks/AgentTuning/WebShop
bash "$W/tests/test_agent_eval.sh"
python -m unittest discover -s "$W/tests" -p 'test_rebuttal_gate.py'
python -m unittest discover -s "$W/tests" -p 'test_rebuttal_baselines.py'
python -m unittest discover -s "$W/tests" -p 'test_rebuttal_aggregator_metrics.py'
```

Results before and after: 36 dry-run launcher rows, 14 Gate tests, 22 baseline
tests and 16 aggregation tests passed. No pre-existing failures were observed in
these selected suites. They are not a claim that every historical repository test
or environment-dependent integration test was run.

New suite:

```bash
python -m unittest discover -s "$W/tests/seek" -p 'test_*.py'
```

Result: **59 tests passed**, final full-suite run 28.615 seconds. The suite covers
strict public/private records and opaque names; source offsets and protected
goal/product/action facts; immutable capture and fresh replay; truncation and
stale-prefix rejection; predicate support and tri-state outcomes; role schema,
refusal/timeout/retry budgets and probe citations; actual likelihood updates and
k=2 fixtures; task/product split dependence; independent outcome-blind holdouts;
frozen candidates and registered families; confirmation bounds and invalid arms;
all five method switches; interrupted resumption and row locks; fake/real
segregation; missing/inconclusive aggregation; clean-control blockers; signature
false-match revocation; dry-run shell paths with spaces and Slurm spool copies.

Additional successful checks:

```bash
python -m py_compile seek_eval.py docs/seek/prepare_cluster.py "$W"/seek/*.py
bash -n seek_eval.sh seek_submit.sh
python seek_eval.py --help
python docs/seek/prepare_cluster.py --help
python seek_eval.py preflight --config configs/seek/fake_cpu.json --metadata-only
bash seek_submit.sh --config configs/seek/cluster_pilot.json --phase collect --dry-run
git diff --check
git diff --exit-code -- agent_eval.sh "$W/test.py" "$W/defenses" \
  REBUTTAL_EXPERIMENTS.md train_fastchat.sh train_lora.sh
```

Every configuration template loads through the strict config validator. Real
metadata preflight correctly exits 2, enumerating absent asset/task manifests,
checkpoint/config files and checkpoint/environment fingerprints. Clean control
remains disabled. Neither metadata validation nor a dry-run is GPU verification.

The exact CLI sequence `collect -> replay -> discover -> confirm -> reuse` was
also run with `configs/seek/fake_cpu.json`, followed by `status`, `aggregate` and
`export-results`. The final output is under:

```
/tmp/seek-final-smoke-d846j275/
/tmp/seek-final-smoke-d846j275/seek-review.zip
```

The repository retains the small, explicitly simulated example in
`SIMULATED_SMOKE.json`. It reports 20 captured tasks, 20 valid fake replays, one
discovered fixture candidate, 8 removal and 8 insertion pairs, and **inconclusive**
confirmation. At mean difference 1, each LCB is 0.03967720868007929, below 0.20.
There are no paper rows or validated signatures. The export contains only an
allowlisted review JSON and README; complete prompts/private records are excluded.

Final smoke source hash:

```
d34dba6dacb30116dc9e22fbf151f1ce4d5061f0a628f137faae06781ba6153e
```

Protected Stage I hashes match the values in `REPOSITORY_AUDIT.md`; tracked Stage I
source has no diff. All implementation changes are new Seek entry points, package,
tests, configs and documentation. The original handoff documents were not edited.

Not verified: real checkpoint availability/identity, genuine poisoning or clean
training provenance, exact real model replay, API/model parameter support, Slurm
allocation/conda/GPU execution, real editable source semantics, held-out real
confirmation or transfer. The conservative legacy adapter protects all page text;
its `no_valid_intervention` result must not be reported as a clean-policy finding.
See `PREREQUISITES.md` and `SEEK_RUNBOOK.md` for the next user-launched campaign.
