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

## 2026-10-02: separate legacy content diagnostic

Implemented `seek/content_diagnostic.py`, `docs/seek/diagnose_content.py`,
`seek_content.sh` and `tests/seek/test_content_diagnostic.py`. Instructions and
interpretation are in [CONTENT_DIAGNOSTIC.md](CONTENT_DIAGNOSTIC.md).

Commands run locally:

```bash
python -m unittest discover -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek -p 'test_*.py'
# 115 tests passed, including 15 new simulated content-diagnostic tests.
python -m unittest discover -s agent-backdoor-attacks/AgentTuning/WebShop/tests -p 'test_rebuttal_*.py'
# 52 tests passed.
bash agent-backdoor-attacks/AgentTuning/WebShop/tests/test_agent_eval.sh
# All 36 Slurm dry-run matrix rows passed without API credentials or model calls.
bash -n seek_content.sh
python docs/seek/diagnose_content.py --help
git diff --check
```

No diff in `agent_eval.sh`, WebShop `test.py`, or Stage I defenses. Existing result
files and snapshots were not changed. The new worker dry-run is tested with paths
containing spaces, an unavailable conda source, and synthetic checkpoint metadata;
this establishes launch plumbing only. The CPU fake victim deliberately changes
its action under capitalization to test paired measurement and accounting; that
behavior is not evidence about either trained checkpoint. No local GPU experiment,
model download, paid API call, training, or `sbatch` submission occurred.

The user previously reported successful real v2 no-edit replays and a Qwen role
smoke. This new capitalization intervention has **not** been run on either real
checkpoint. Its eligibility and response differences await the separate cluster
prepare/dry-run/pilot sequence. Training binding and overlap remain limitations,
not prerequisites to implementing or running this exploratory diagnostic.

## 2026-10-06: legacy click normalization and offline rescoring

The diagnostic scorer now matches the environment's lowercased click arguments,
lowercase legal-click keys and case-normalized product-ID bindings, including
`Buy Now`. Raw proposals and the environment itself remain unchanged. Added
`docs/seek/rescore_content.py` for immutable CPU-only rescoring of completed
original runs. Tests cover actual lowercase clickables with uppercase titles,
case-equivalent duplicate ID rejection, purchase normalization, unknown-action
rejection, original-file preservation, reply/plan/snapshot verification and
model-free imports. The focused content suite has 19 passing CPU tests. These
are synthetic fixtures; corrected product metrics on the user's real snapshots
await the cluster CPU rescore, with no new generation required.

Full Seek regression command `python -m unittest discover -s
agent-backdoor-attacks/AgentTuning/WebShop/tests/seek -p 'test_*.py'` passed all
119 tests after this correction. `git diff --check` also passed. Protected
`agent_eval.sh`, `test.py` and `web_agent_text_env.py` have no diff. No GPU jobs,
model downloads, paid APIs or training were run.

### 2026-10-06: native WebShop price ranges

Scorer v3 accepts WebShop's `$low to $high` price fields for title binding. The
focused CPU suite passes 21 tests, including the exact exported Levi's title and
price layout, an Adidas-title positive control, item-page ranges, and malformed
range rejection. All eight result-page observations in the user's earlier source
audit parse to the same Levi's title when supplied a minimal legal-click context;
this is an observation-parser check, not full snapshot or GPU verification.
Original cluster artifacts remain untouched; `rescored_v3.json` is a new output.
