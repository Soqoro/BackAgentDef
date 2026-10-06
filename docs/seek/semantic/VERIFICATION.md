# Local verification, 6 October 2026

Baseline commit: `2dff2f1a423761d391179890af5995e168d9ddc4`. The user-supplied handoff directory and earlier audit reports were already untracked and were preserved. No commit or push was performed.

| Check | Result |
|---|---|
| Full Seek CPU suite, including semantic tests | 182 passed: 139 existing + 43 new |
| Selected Stage I rebuttal regression suite | 52 passed |
| Existing `test_agent_eval.sh` launcher dry run | 36 rows passed, no Python model invocation or API credentials |
| Semantic worker dry run | Help path passed with unavailable asset paths; no conda/model load |
| Python compilation, shell syntax, runbook shell syntax, `git diff --check` | Passed |
| Protected-file hashes before/after | All 26 unchanged |

Commands are in [RUNBOOK.md](RUNBOOK.md). Raw test output is retained in `test_output.txt`. The semantic tests cover numerical equivalence to the supplied independent reference, strict thresholding, unknown/positive eta, monotonic penalties, zero evidence, global/reserved/atomic j allocation, mutation detection, review binding, exposure rejection, IID-only design, outcome-conditioned eligibility rejection, semantic/lexical distinction, persistent observation slots, parsing/negation/ambiguity, live RNG and stub action isolation, incomplete/malformed pairs, deterministic and stochastic resume, partial pairs, adaptive new claims on fresh support, raw-ledger export reconstruction, public role contexts, bounded citation failures, current config compatibility, metadata imports and a small known-rule suite.

Protected hashes still match the prior audit:

- `agent_eval.sh`: `7920d22628ec0ed620bc4e0a762aaedf39be7c891e598c8cb000c84613dd22d5`
- `agent-backdoor-attacks/AgentTuning/WebShop/test.py`: `d6eb0b9665f5b8554a182c00bbf196c553f835f6773068998aceae840d1dd766`

The only modified pre-existing source is `seek/local_roles.py`: an overridable message builder and separate startup timing. Its default builder still invokes the original role protocol; existing regression tests pass. All semantic functionality is namespaced in added files.

## Shipped evidence

- `budget_j1.json`: algebraic budget, not observed results.
- `local_status.json`: no native plan/results available locally; pending with null measurements.
- `simulated_role_transcript.json`: explicitly scripted fake roles, four nonempty replies. No Qwen validation.
- `simulated_smoke.json`: one seeded study, two known fake policies, four paired draws each. Both inconclusive at this deliberately tiny budget; null mean 0, concept mean 1. Zero false certificates in one replication provides essentially no empirical precision about rare error rates; Wilson uncertainty is exported.
- `simulated_export/report.json` and `.html`: consistent contract/result audit, clearly simulated.
- `simulated_raw_pairs.json`: raw replies, consumed tokens and paired draw records corresponding to that export, permitting bound reconstruction. Fixture timestamps and measured latencies are not claimed byte-deterministic; inputs, outputs and scores are seeded/reproducible.

No real semantic effect, Qwen role quality, fresh victim replay, observation opportunity, cluster runtime, memory fit or new model behavior was verified locally. The earlier real smokes and wording studies remain historical evidence. No GPU jobs, Slurm submissions, model downloads, paid APIs or training were run. The first new real milestone remains the reviewed nonempty query pilot in the runbook.

## File map

| File(s) | Purpose |
|---|---|
| `seek/semantic_stats.py` | Exact manuscript confidence sequence and separate budget/detectability calculations |
| `seek/semantic_contracts.py`, `semantic_registry.py` | Strict contracts/reviews, persistent family/index allocation, exposure reservation and append-only audit |
| `seek/semantic_renderers.py` | Audited coherent paired inputs and conservative parsed-action scoring |
| `seek/semantic_runner.py` | Stateless paired calls, input checks, immutable evidence, missingness and resume |
| `seek/semantic_roles.py` | Separate Action/Goal/State protocol, source validation and bounded finite-library scheduling |
| `seek/semantic_evidence.py` | Native/historical import, exposure lineage, raw-result verification and JSON/HTML export |
| `seek/semantic_simulation.py` | Known-rule CPU policies and study-wise Monte Carlo reporting |
| `seek/semantic_cli.py`, `seek_semantic.py`, `seek_semantic.sh` | Metadata, compilation, review, registration, pilots/discovery and confirmation recipes |
| `tests/seek/test_semantic.py` | 43 CPU tests; no real-model claims |
| `configs/seek/semantic/` | Actual implemented schemas, simulated draft, explicitly evaluator-specified query/observation examples |

The optional omitted-semantic-review real ablation remains outside the initial exposed CLI; adaptive, fixed-order and discussion-only are implemented. No live repair or new end-to-end Shield gain is claimed. Large simulation studies and all new real cluster experiments are user-run recipes, not completed experiments.
