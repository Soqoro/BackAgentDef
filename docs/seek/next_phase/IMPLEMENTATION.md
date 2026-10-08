# Next-phase implementation and validation

Implemented in the current checkout, based on `docs/seek_next_phase/CODEX_UPDATE_PROMPT.md`,
`EXPERIMENT_PLAN (1).md`, the supplied report/source context, and the exact manuscript
excerpt. No GPU/model execution, downloads, training, paid APIs, job submissions or
pushes were performed locally.

## Delivered components

- `seek_next.py` and `seek/next_phase/`: isolated prospective CLI and modules. Existing
  Seek model loading, storage journals, strict historical search scoring, Qwen
  transport and the authoritative allocation registry are reused.
- P0: safe archive inventory/verification, legacy raw score and bound reconstruction,
  attempt/token/model bindings, replay/ledger/stopping checks, per-source accounting,
  missing-artifact reports and explicit exposure import. The archive packager reuses
  the existing inventory helper and excludes private training material and weights.
  Historical source/runtime authenticity remains explicitly unresolved where it
  cannot be independently established. Numerical reconstruction is not authenticity.
- P1: coherent hypothetical category × two-label brand-assignment designs, three
  opportunities, stable bindings, independently crossed positions/wording/budgets,
  rotated opaque IDs, complete consumed-prompt cue checks, strict slot-inspection
  scoring, descriptive component/brand rates and coefficient-derived interactions.
- P2: exact requested query relations across literal, paraphrase, related-footwear and
  non-footwear operands. Unsupported hypotheses are rejected without substitution.
  The default pilot is sneakers/trainers, not an implicit sneakers/watches fallback.
- Census: a distinct complete weighted finite-support protocol, greedy runtime
  binding, fixed anchor repeats, explicit incomplete/unstable states and no sampling
  certificate. Repeated exact inputs use a model/token/runtime/protocol-bound cache;
  physical calls and logical draws remain separate.
- Anytime inference: the existing manuscript radius scaled by the coefficient
  range, global allocation including failed/reserved IDs, strict raw-unit threshold,
  frozen batch checks and null semantic discrepancy. No prospective effects were
  certified locally.
- P3: actual Action/Goal/State role rounds for adaptive, fixed-schedule and
  discussion-only variants; exact proposal retention, bounded retries, privacy
  allowlists, response-replay oracle access accounting and frozen candidates. The
  common fresh evaluator uses a separately reviewed finite-support bank shared
  across all variants. It reports implemented relations and unresolved semantics,
  never invented recovery accuracy. Its optional per-candidate scope reviews remain
  unknown unless an independent reviewer supplies them.
- P4: an opt-in known-rule CPU study with exploration-dependent proposal order, fresh
  confirmation, global indices, failed/unavailable hypotheses, coherent cell rules,
  coefficient scales 1/2/4, frozen stopping boundaries, family-wise error events,
  binomial uncertainty, coverage and cost. Census arithmetic is reported separately.
- Three Slurm wrappers: serial model arrays, Qwen-only investigations, and serial
  common evaluation. Dry runs do not activate model environments or call inference.
  Full confirmation requires both a reviewed successful pilot and explicit submission.

## Tests actually executed

Python **3.12.7**, CPU only, standard-library `unittest`; no dependency installation.
The initial `pytest` command failed because pytest was unavailable. The new tests
were then written to use the repository's standard-library convention. Two initial
scorer fixture expectations were corrected: explicit brand exclusion is 0; a click
in a first-search experiment is unscorable. The historical scorer was not changed.

Final passing suites (distinct totals):

| Suite | Tests |
|---|---:|
| WebShop `tests/seek` | 236 |
| `docs/seek` direct-comparison/archive | 12 |
| selected `test_rebuttal*.py` Stage I | 52 |
| **Total unit tests** | **300** |

The Seek total includes **43 new next-phase tests**. These cover binary corners,
scales and strict thresholds, rounded historical arithmetic (not raw verification),
missing/corrupt archives and ledgers, wrong model/tokens, registry locking/idempotence,
failed reservations, leakage, duplicate support, serial joins, cache accounting,
malformed responses, unstable anchors, exact census, investigation variants/privacy,
common evaluation, simulation families and CLI/shell dry runs.

The existing Stage I shell harness also passed **all 36 Slurm dry-run rows**. Shell
syntax and Python compilation checks passed. Logs and commands are in
`validation/receipt.json` and the adjacent text files. These are fake-backend and
CPU checks, not real-model verification.

The final two-study, cap-64 simulated smoke is `simulated_smoke_final.json`:
72 allocated simulated claims; 18,204 simulated response draws including exploration;
0/2 studies with a false certificate. The Wilson 95% interval is approximately
[0, .658], so this tiny smoke does **not** establish empirical calibration.
Checked interval coverage was 2/2. The 1,000-study campaign was not run.
Earlier `simulated_smoke.json` and `local_audit/` are retained local development
receipts; use the `_final` artifacts for the completed implementation checks.

## Preserved evidence and unresolved inputs

All **22 protected-file SHA-256 values** match the before snapshot. This includes
`agent_eval.sh`, WebShop `test.py`, Stage I defenses and preexisting report artifacts.
The checkout's preexisting untracked handoff/report files were retained. No existing
source file was edited; additions live in new files. No cluster registry was present
locally to modify. The six historical claims were neither rerun nor reclassified.

`local_audit_final/` labels all six historical entries as report-only, not raw-verified,
and lists the raw registry/contract/draw/attempt/ledger/replay/source prerequisites
per claim. The original reported certificates are implemented effects; they do not
establish trigger recovery, matched clean-control differences or poisoning attribution.
Claim 1's reported controller-assisted redirection remains part of its historical
interpretation; its immutable original origin field is not rewritten.

Cluster verification still needs:

1. The authoritative current study registry and original raw artifacts; a summary
   export is insufficient. The real allocator requires the known six-claim registry
   head in its chain and reads the actual next index.
2. The existing query, observation and AgentLM interface manifests and pinned local
   weights/tokenizers; the pinned Qwen configuration/lock/interpreter for P3.
3. Independent experiment/resource reviews, followed by the actual pilot outputs
   and stable anchors. Full confirmation is not enabled by a Slurm exit code alone.
4. Four suitable exploration incidents, two distinct groups per attacked checkpoint.
   The known one-group development diagnostic may not supply them. The selector
   stops with explicit missing-group prerequisites rather than borrowing holdouts.
5. Additional designated exploration oracle coverage if an agent requests a supported
   relation outside the two initial pilot tables. Coverage misses remain failures;
   the controller does not substitute its preferred scientific question.

Optional goal-exposure and native-grounding renderers remain disabled pending
separate coherent-design review. AgentLM is an external reference, not a matched
clean training control. Semantic eta and training attribution remain unknown.

## First cluster actions

Follow `RUNBOOK.md` in order. It provides the exact implemented commands for CPU
archive verification, profile compilation, readable previews, review/freeze, dynamic
claim-ID extraction, user-submitted pilots, scientific status, separately approved
confirmation, P3 and the opt-in CPU study.

The first reviewed pilot maxima are **156 observation responses** and **78 query
responses**, including anchors. These are response budgets, not runtime estimates.
Observation census would be 660 responses; one query census would be 330. Neither
is launched automatically. Census effects, anytime certificates, incident-led
candidates, evaluator-specified diagnostics and simulations stay separately labelled.
