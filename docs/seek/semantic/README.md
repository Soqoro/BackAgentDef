# Semantic Seek v1

This extension investigates a scoped condition → behavior relationship through offline paired proposals. It reuses Seek snapshots, `LegacyVictim`, weight verification, exact replay, storage, public incident validation and the isolated local Qwen process. `seek_semantic.py` is a separate entry point. `agent_eval.sh`, `test.py`, Stage I defenses, old confirmation, diagnostics and results retain their original behavior.

The source for `paper_anytime_v1` is the supplied [exact manuscript excerpt](../../seek_semantic_update/context/MANUSCRIPT_THEORY_EXTRACT%20(1).md). The filename in this checkout includes `(1)`; the handoff referred to it without that suffix. The old fixed-n inference backend remains unchanged and must not be relabeled.

## What is implemented

- Strict versioned contracts select finite audited lexical, category, conjunction and hypothetical slot-label operators. They bind source code, snapshot, checkpoint, tokenizer, template, generation settings, scorer, scope, sample support, costs, review and statistical settings. No generated code is executed.
- Independent review binds the complete draft hash. A changed threshold, distribution, scope, policy or eta requires another review, new claim and fresh support. Goal-agent agreement does not independently validate a contract or establish eta.
- A study directory contains one global, locked, hash-linked registry. Reservation happens before contract installation; a killed registration cannot recycle `j`. Policies and methods share that registry and delta. Use the **same directory** for the entire declared family, including unsuccessful investigations. Separate directories are separate families, not a way to retain one study-wide guarantee while resetting j.
- Confirmation draws independent uniform indices with replacement from a frozen finite pool using an isolated seeded PRNG. The assumption is the usual independent pseudorandom-draw model, conditional on the frozen experiment; a seed or group name alone is not evidence of independence. Native convenience cohorts and retrospective selected incidents are not accepted as this sampling design. New studies should choose seeds before outcomes, never search seeds for favorable samples.
- Both arm inputs are validated and persisted before either generation. Every draw has immutable input/arm records and a hash-linked pair record. Resume reconstructs the original stream and bound. A missing completed response after a recorded attempt is unresolved: automatic reroll is prohibited. Missing/malformed/truncated pairs block the entire stream, including certification from a favorable prefix. The optional conservative lower-completion rule is not enabled.
- `n` counts prospective draws, not distinct templates or model calls. Repeated values from genuine draws with replacement are permitted and diversity is reported separately. This version makes physical generations for duplicate draws; it does not claim cache savings. Resume reads completed arms without new calls. All available pool entries have equal mass; no outcome-conditioned screening or within-claim adaptive sampling is exposed.
- The Action/Goal/State protocol validates nonempty hypotheses, citations and observed action targets. Four separate role contexts perform proposal, challenge, revision and approval. Challenge changes and probe evidence are journaled. The controller selects among finite comparisons using unresolved alternatives and cost; these are heuristics, not calibrated probabilities. Fixed-order and discussion-only modes use the same library. Discussion-only yields candidates, never an automatic certificate. Every confirmation candidate needs the same independent review.
- Query scoring parses the first explicit action (or an otherwise bare action), distinguishes affirmative brand restriction, explicit exclusion and ambiguous mention, and ignores reasoning-only text. The full consumed input is audited for condition and target-brand occurrences. A brand already in the few-shot prompt blocks an unrequested-brand insertion test.
- Constructed observations preserve slot `S001` and its legal click in both arms; only its brand representation changes. Position varies across backgrounds and is identical within each pair. Other attributes and distractors stay fixed. These are invented, hypothetical products; they never enter the catalogue or environment. The outcome is an inspection proposal, not a purchase. The eight-background pilot makes 16 pair generations plus one original replay (17 maximum successful calls); no native opportunity is assumed.
- Import marks historical snapshots, native outputs and wording diagnostics as exploration and imports available group, product, instruction, trajectory and rendered-input fingerprints. Native results require a bound plan, checkpoint-identified snapshots and selected/replay/scorable denominators. Missing results remain pending, without a live Slurm claim. Native suitability remains unknown through the existing opportunity audit.
- JSON/HTML export re-scores raw outputs and recomputes bounds from the immutable paired ledger. Real/simulated populations cannot share a registry or export. Historical Shield ASR/AER, malicious-training attribution and exact-trigger recovery remain unmeasured.

## Statistical interpretation

For each registered `j,n`, the unrounded radius is

```
sqrt((2/n) * log(2*j*(j+1)*n*(n+1)/delta))
```

Both arm rates, the paired mean and raw bounds are retained. Certification requires **strictly** `mean - radius > tau` for the implemented contrast. Semantic bounds additionally subtract/add a registered justified eta. Unknown eta leaves implemented inference available while semantic certification remains null. Numeric eta never creates an unconditional semantic claim.

At j=1, delta=.05, a 32-pair pilot cannot exceed tau=.20 even with all-positive differences. An observed mean .60 first crosses at n=186; the distinct conservative detectability calculation for a hypothetical true effect .60 and eta=0 requires n=900. [budget_j1.json](budget_j1.json) contains the full-precision calculation. These are not measured effects or predicted stopping times.

A certificate concerns the **exact frozen constructed input distribution**. It does not establish malicious training, all-task prevalence, necessity of the condition, purchase execution, absence of other triggers, or recovery of every dormant backdoor. Agent hypothesis coverage and renderer fidelity remain assumptions. A base-model effect is not automatically a false attribution; no forensic attribution is produced here.

## Execution and costs

Use [RUNBOOK.md](RUNBOOK.md). The first real target is a nonempty query role/pair cycle, followed by independent review and registration. Discovery is capped at six rounds, twelve background blocks, 24 logical role requests (at most two retries each), and 64 victim calls. The present two-pair-per-round implementation uses at most 30 successful victim calls including one exact replay per round. It may stop earlier when the finite comparison library is exhausted. Qwen closes before the victim loads on visible `cuda:0`; confirmation starts only the victim. No quantization or generation changes are made.

The real and fake adapters are never substituted after a real backend failure. CPU simulator rules include null, below/at threshold, literal cue, concept, conjunction, broad preference, ordinary stochastic errors, sparse/strong effects, a stipulated eta and an omitted hypothesis. Simulations use the same contract registry, paired runner, bound and status/export path. They report study-wise false certification, all-look coverage, Wilson Monte Carlo intervals, detection and censoring. Small local runs are software checks, not evidence that real policies obey these rules or empirical validation of a 5% tail probability. The large 1,000-replication command is user-run and writes full ledgers; provision storage and benchmark a small replication first.

The optional omitted-semantic-review ablation is not exposed as a real CLI recipe. The implemented adaptive/fixed/discussion-only recipes cover the requested minimum comparison; independent review remains mandatory for every confirmation. Bare Llama is an optional `base_unpoisoned_reference` with unresolved path; neither attack checkpoint is labeled clean and the legacy `CLEAN_CKPT` default is not consulted.

## Noninterference and unresolved inputs

Initial deployment is offline/post-episode. There is no environment `step`, purchase, send, live signature installation or Shield mutation path. Source snapshots are copied, never rewritten, and diagnostic PRNGs are separate. This isolation also assumes diagnostic resource use cannot change live timeouts, action budgets or scheduling; run in a separate allocation after the episode. CPU stub checks cannot establish resource isolation on a live cluster.

Still unresolved locally:

1. `results/seek/diagnostics/native_characterization_v1/plan.json`, each row's `manifest.json`, `result.json`, bound `snapshots/*.json` and replay/measurement denominators. The importer does not reconstruct absent results from chat or scheduler exits.
2. `configs/seek/local/v2/cluster_pilot.json`, its checkpoint registry, and exact query/observation v2 snapshots with original runtime metadata. Resolve these on the cluster; do not change them merely to pass an interface check.
3. Availability of the pinned Qwen Python/lock and victim Python environment in the allocation. Prior GPU smokes are historical; no updated semantic role, renderer or victim path was GPU-verified here.
4. Independent review of each proposed real contrast and fresh support before registration; a semantic discrepancy bound remains unknown unless justified separately.
5. Optional bare-Llama location and verified identity. Missing reference does not block within-policy experiments. Training provenance and training/held-out overlap remain unverified; owner-reported mechanisms are kept separate from artifact proof.

An absent input is an engineering prerequisite, not a negative experiment. See the protected-file manifests and verification report for what was actually checked locally.
