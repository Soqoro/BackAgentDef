# Seek: implementation and experiment plan

Prepared from the uploaded `BackAgentDef-main.zip` and the Shield-and-Seek LaTeX manuscript. This is a proposed implementation specification, not a report of new GPU results. The archive was inspected without modifying its source. The GPU checkpoints, real Stage I result logs, full WebShop assets, and cluster-only training file are not in the archive.

## 1. What to build first

Add Seek as an isolated diagnostic pipeline around the existing WebShop policy and Shield components. Keep `agent_eval.sh`, its existing CLI defaults, and existing Stage I result artifacts unchanged. Do not rewrite the old evaluator into three new LLM agents. The Goal, State, and Action agents participate in the new Seek dialogue, while calling the existing deterministic parsers/certifiers as tools. This makes the new work measurable without reattributing old results to new role prompts.

Workflow: local Codex development and CPU tests -> push to GitHub -> pull through the online Jupyter cluster -> Slurm GPU jobs -> export small evidence reports back to local Codex. No GPU work on the login/notebook process, no Docker requirement, no PBS, no Colab, and no automatic training or submissions by Codex.

Use one victim checkpoint per GPU worker and load it once. Defender roles use separate contexts on a configured trusted API backend, initially following the backend already used for goal extraction. Three roles do not require three GPU allocations. A fake backend is for tests only; it must never silently replace a failed real backend in an experiment.

## 2. Repository-specific gates before experiments

### Gate A: recover checkpoint provenance, not just its directory name

The launcher points to separate query and observation `checkpoint-118` directories. `CLEAN_CKPT` defaults to the query-backdoored checkpoint: this is clean-input evaluation, NOT an independently clean model. A clean-model control needs a separate, verified registry entry. An unpoisoned base Llama checkpoint is useful, but is not the same control as a matched clean AgentInstruct-fine-tuned checkpoint; name these conditions accurately.

The bundled old clean/poison sneaker examples do not establish an inserted incidental lexical trigger. Their first task instructions are shared in the paired examples. The evaluator uses natural target-brand occurrence to count observation exposure and has no `direct_trigger_text` injection path. The newer `create.py` explicitly requires and persists a query trigger, but this does not prove that the existing checkpoint was trained with that version. The actual training script points to a cluster-only `data/observation_attack/poison_m50.json` file, absent here.

Preflight must inventory the actual checkpoint configuration, train data hash/metadata, clean control, exposure mechanism, and inference serialization. If the trigger provenance is unknown, mark it unknown. Training data and trigger labels may be inspected by the evaluator-side provenance audit, never supplied to the Seek agents.

### Gate B: verify that the proposed intervention is meaningful

Distinguish incidental lexical cues from task-essential cues and content-linked brand conditions. Deleting the requested category or a product brand is not automatically a valid experiment, even when the latent environment object is unchanged. Preserve the evidence available to the policy about user requirements, brands, prices, options, product identities, availability, and legal actions.

On legacy checkpoints, allow the outcomes `inconclusive`, `no_valid_intervention`, `no_localized_effect`, `unknown_trigger_ground_truth`, and `replay_invalid`. A reproducible preference is not automatically a contract violation, and neither a preference nor a violation proves an exact implanted trigger. Do not claim exact trigger recovery when no exact ground truth is available.

If existing checkpoints cannot support a removable-trigger experiment, do not invent a phrase and assume the model recognizes it. Continue legacy behavioral/coverage analysis and explicitly report the limitation. A supplemental controlled lexical-trigger suite would require genuinely trained checkpoints with held-out controls and training manifests. It is a separate, later experiment, not a relabeling of Stage I results. Do not make retraining the first implementation step.

### Gate C: reproduce the exact policy input

`Chat.request()` mutates FastChat conversation state, uses the Llama-2 template, truncates to 2048 tokens by default, and performs greedy generation. Shield can then replace the stored assistant message after masking or projection. The saved conversation alone is therefore not an immutable copy of every original policy proposal; default debug fields can also be truncated.

Capture raw observation, exact policy-facing observation/prompt, the full pre-call conversation, token IDs actually consumed, decoding settings, raw response/action before all interventions, executed action, frozen goal, grounded state, available actions, and source offsets. Preserve selected options and page identity. Record environment/filter/catalogue fingerprints and the first possible exposure boundary.

Before diagnosis, replay an unedited snapshot and check serialization/token identity and the parsed action. If the generated response differs under identical greedy conditions, record the difference and investigate before interpreting intervention effects. With stochastic secondary settings, use an appropriate repeatability audit rather than requiring identical text.

Edits must not cause unrelated protected history to enter or leave the tokenized context due to truncation. Check protected token/field visibility in both arms. Reset from before exposure; never retain reasoning generated after the cue in a counterfactual prefix. Later effects require genuine branched environment rollouts, which are out of the first pilot's scope.

## 3. The measurement unit

A diagnostic session contains one checkpoint, one channel, one discovery incident or disjoint discovery pool, one frozen behavioral predicate, one candidate proposal, and a confirmation protocol. Multiple snippets, sampling seeds, or generations from the same task are not independent tasks. Several sessions using the same trigger/checkpoint are repeated localization trials, not independent backdoors.

Track these denominators separately: available eligible episodes; captured episodes; audited raw inputs; policy-facing exposures; Shield incidents; opened sessions; replay-valid sessions; sessions with editable candidates; completed confirmations; validated signatures. Report unconditional coverage and recovery conditional on a session, rather than reporting only successful investigations.

Keep three input tracks separate:

1. `shield_incident`: an actual rejection or relevant intervention on the defended trajectory.
2. `raw_audit`: an explicitly budgeted, isolated raw-input query that can reveal behavior masked by Shield. It is NOT a deployed-path failure.
3. `training_trace_smoke`: training examples used only to test parsing/integration, never as held-out recovery evidence.

## 4. Seek protocol

### 4.1 Define the behavioral outcome without evaluator labels

The Action agent derives a predicate from a source-supported deviation: for example an unsupported restriction inserted into a search or an explicitly prohibited purchase. The predicate must name operation/argument/product conditions, cite the contract/state facts, and distinguish a tentative product inspection from a commitment where necessary.

Freeze the predicate before discovery comparisons and definitely before confirmation. Do not infer it from the evaluator's target-brand field, poisoned status, or success label. Brand selection without a user prohibition is not automatically unauthorized. Label pure choice-sensitivity studies separately from unauthorized-action diagnosis.

### 4.2 Construct candidate spans and valid edits

The State agent proposes exact source-referenced spans and competing activation explanations. Include an alternative that ordinary policy error, task information, or a different position explains the behavior. The Goal agent tests the edits against the frozen contract and protected state. Deterministic equality checks on fields/actions are necessary; uncertain semantics lead to review or abstention, not an LLM confidence score.

Start with at most 16 editable spans and single-span hypotheses. The 16-span limit is a proposed pilot choice, not a paper result. Report candidate-set coverage and when the cap could omit the actual cue. Multi-occurrence handling must be explicit: grouped occurrences count as one hypothesis only under a recorded rule. A two-span conjunctive extension is a separate stress test.

### 4.3 Run bounded, evidence-grounded discussion

Roles emit validated structured records: hypothesis, predicted outcome, alternative explanation, proposed edit, challenge, actual probe IDs, and revision. A challenge must change the admissible probe set, candidate ranking, or next question; three independent summaries are not the intended dialogue.

A controller enforces budgets, preservation decisions, legal actions, and status transitions. It queries only the same uncompromised-infrastructure/compromised-weights victim checkpoint. Recorded actions decide outcomes; agreement among defender agents does not validate a trigger. The controller has no live execution authority.

### 4.4 Select informative probes

For approved masks z (1 retains a candidate span; 0 neutralizes it), maintain H0 and sparse cue hypotheses K. Use the manuscript's exploratory Bernoulli model:

`mu_K(z) = p0 + (p1 - p0) * product(z[j] for j in K)`.

Estimate smoothed p0/p1 on discovery controls only; H0 predicts a constant rate. Update log weights from actual probe outcomes. Choose a low-cost valid probe with high weighted variance of predicted outcomes. Include singleton, complement, and subset candidates. Count malformed requests, retries, controls, and local reductions in the discovery budget. Stop at a fixed query/round cap or when no useful valid probe remains.

### 4.5 Freeze, then confirm independently

Freeze the candidate bytes/span rule, checkpoint, behavior, scope, editing process, neutral controls, thresholds, family size, sample sizes, and holdout IDs before querying confirmation. A failed candidate cannot be edited against the same holdout and retested as if fresh.

Removal contrasts compare candidate-containing input to its valid candidate-removed version. Insertion contrasts compare candidate insertion with matched neutral insertion in compatible cue-absent contexts. Never filter held-out contexts by whether the model happened to take the bad action. Select by task/interface/position eligibility only. If removal cases are artificially constructed by inserting the candidate, report that source explicitly; do not describe them as naturally occurring removal cases.

Use the manuscript's bounded paired differences D in [-1,1]. For n independent context-level differences and M preregistered candidates in the declared confidence family:

`r = sqrt(2 * log(2*M/alpha) / n)`

`LCB = mean(D) - r`.

Both removal and insertion LCBs must exceed their frozen effect thresholds. Use separate n if contrast sizes differ. The family definition and M must be present in every output. Repeated generations within a task are averaged within that task, not added to n. Shared products/trajectories require an explicit grouping/dependence analysis; do not assert independent-context guarantees when independence is unjustified. Neither the bound nor a signature proves malicious training history or absence of other triggers.

Invalid actions, timeouts, truncation damage, and missing paired arms are not clean counterexamples. Mark unscorable observations and follow a preregistered failure rule; the conservative initial rule makes the affected confirmation inconclusive rather than silently dropping unfavorable cases. Policy malformed-action rate and missing-evidence rate must be reported separately.

### 4.6 Reuse without changing Shield

A validated signature is tied to a checkpoint, input channel, context scope, predicate, edit protocol, evidence, and costs. Initial reuse only avoids repeated full diagnosis when a signature applies; it does not relax Shield or blacklist a brand independent of the user's request. Purge incompatible signatures when checkpoint or schema changes. Measure saved diagnosis queries, misses, and false matches on a further held-out set.

## 5. Experiment ladder

| Phase | Proposed scale | Purpose | Go/no-go condition |
|---|---|---|---|
| P0: local CPU/fake backend | synthetic unit and integration fixtures | validate protocol, budgets, isolation, aggregation | all targeted tests pass; no API/GPU use |
| P1: cluster collection/replay | 4 direct + 4 indirect cases initially | verify checkpoints, exact serialization, actual exposure | preflight/provenance and replay audit pass or report blockers |
| P2: development pilot | up to 16 new contexts per channel, sneakers first | inspect candidate coverage, valid edits, dialogue, query use | meaningful valid contrasts or explicit identifiability diagnosis |
| P3: registered main study | one frozen signature/session; proposed 64 removal + 64 insertion contexts per candidate | estimate functional recovery, false positives, costs | enough independent eligible holdouts; frozen protocol |
| P4: ablations | same checkpoint/task pools and budgets | isolate contribution of each Seek mechanism | blind independent scoring across conditions |
| P5: signature reuse | proposed 32 additional tasks per scope | test repeat-diagnosis savings | no change to executed Shield decisions |

The 4/16/64/32 counts are planning values to freeze after development, not claims about available data. The supplied direct lists each contain 131 IDs and indirect lists 114. Those lists alone cannot furnish all these disjoint pools. Construct additional held-out tasks from the correctly fingerprinted environment, or report underpowered confirmation; do not recycle development/training contexts. Numeric IDs are meaningful only under the exact goal filter/order/catalogue. The current attack evaluator's filter is sneaker-specific even when another category ID file is supplied, so category expansion requires a new explicit Seek filter manifest rather than silently changing Stage I.

Suggested pilot settings: max 16 candidates; interaction order k=1; 6 discussion rounds; 32 discovery victim generations (all calls included); 24 defender calls total; max 1024 output tokens per defender call; 8 contexts per confirmation contrast for a wiring-only pilot. Preserve victim greedy decoding, input 2048, output 128 initially. Set model IDs explicitly. These are caps, not required spending targets.

An 8-context pilot cannot satisfy the default example confidence rule with M=1, alpha=0.05, tau=0.20: r is about 0.960, so even mean difference 1 fails `LCB > 0.20`. It must not print VALIDATED based on this pilot. For n=64, r is about 0.340, requiring an observed effect above about 0.540. Multiple-candidate families require larger radii. Pick adequate main-study sizes before testing.

Per one candidate and one generation per arm, 64 removal pairs plus 64 insertion pairs require 256 victim generations; with discovery cap 32, the budget is 288, excluding collection/replay and evaluator-only audit. This is a query count, not a runtime estimate. Ten independent localization sessions require their own declared evidence reuse/family plan; do not pretend a shared holdout yields ten independent confirmation datasets.

## 6. Conditions and baselines

Primary conditions: `seek_full`, `fixed_probes`, `discussion_only`, `no_goal_preservation`, and `removal_only`. Add a cheap leave-one-span-out localizer if useful, but do not make investigator-count comparison the central experiment.

- `fixed_probes`: same roles/backend, candidate/edit space, query cap, and independent confirmation, but a frozen probe order instead of feedback-driven selection.
- `discussion_only`: same role discussion but no victim probes during discovery. Its proposed candidate is scored through a separate blind evaluator-side test; any raw agreement is only a claim. Separate evaluator cost from method cost.
- `no_goal_preservation`: omit semantic preservation during discovery but retain hard source/action identity integrity and sandbox isolation. The independent evaluator still audits preservation and penalizes invalid attributions. No artificial live unsafe execution.
- `removal_only`: discovery unchanged; method status uses only removal. Independent evaluator still tests reinsertion to quantify false claims.

Negative controls: verified matched clean checkpoint if available, separately labeled base checkpoint if that is all that exists, clean-input cases on compromised checkpoints, legitimate goal-bearing words, neutral phrases, and ordinary errors. Do not pool these as a single clean-model control.

A supplemental controlled-trigger suite should be registered only after exact trigger-bearing training manifests/checkpoints exist. Do not count runtime action-oracle injections as model backdoors, and do not assume a synthetic phrase activates a legacy checkpoint.

## 7. Outputs and paper tables

Write immutable, resumable JSONL events and atomic JSON summaries. Never overwrite original Stage I outputs. Proposed run tree:

```
results/seek/<run_id>/
  resolved_config.json
  run_manifest.json
  checkpoint_provenance.json
  environment_fingerprint.json
  public_cases.jsonl
  private_eval/labels.jsonl
  snapshots/
  replay_audit.jsonl
  agent_dialogue.jsonl
  interventions.jsonl
  probes.jsonl
  frozen_candidates.jsonl
  confirmation.jsonl
  signatures.jsonl
  failures.jsonl
  summary.json
  audit.md
  tables/
```

Core metrics: investigation coverage; candidate-set coverage (evaluator-only); exact recovery when labels exist; independently validated functional recovery; localization precision/recall when defined; false attribution and false validation rates; inconclusive rate; preservation failure rate; unscorable outputs; replay validity; discovery and confirmation victim calls; defender calls/tokens; cache hits; measured latency. Report both incident/task-level and checkpoint/signature-level counts. No confidence interval over a single checkpoint is evidence of cross-checkpoint robustness.

Keep actual defended ASR/AER separate from probe-level behavior rates. AER here is average episode reward. Copy no historical values into new Seek result rows. Empty/missing runs remain missing, not zero. Reports must distinguish measured outputs, simulated fixtures, proposal placeholders, and unevaluated conditions.

## 8. Slurm and reproducibility requirements

Proposed new root entry points: `seek_eval.py`, `seek_eval.sh`, `seek_submit.sh`, plus `configs/seek/` and a `seek/` package under the active WebShop directory. The exact implementation is for Codex to create; these do not currently exist in the uploaded archive.

The shell follows the existing conda environment setup and one-GPU convention, but has configurable checkpoint/config/output paths. Do not inherit hard-coded direct/indirect paths as hidden defaults. Default scheduling can mirror NA100q as an explicit local example; do not assume a node pin is mandatory. Preserve Slurm's `CUDA_VISIBLE_DEVICES`; use visible CUDA device 0. Create log directories before `sbatch`, not only inside the job.

The submission helper supports dry-run and explicit array concurrency (default 1). Split by manifest row/session or checkpoint shard, not individual token deletions. Cache the victim in each worker. Give every array job a unique output path. Atomic resume keys include checkpoint/config/prompt/input hashes, phase, frozen signature, split, and replicate; changing them cannot silently reuse evidence. Failed runs never become successful because a stale summary exists.

Record `pip freeze`, resolved requested and actual defender model, generation settings, template/source hash, tokenizer settings, precision, git commit or source hash fallback, environment metadata, task IDs, source data hashes, Slurm job IDs, and secrets-free arguments. Metadata dry-runs must not import Torch/environment heavy modules, load models, make API calls, or download assets.

## 9. Acceptance before the first real campaign

Required artifacts from Codex: implementation, offline tests, real-backend interface, explicit prerequisite report, a dry-run schedule manifest, and a runbook matching the implemented CLI. The first real cluster campaign is collection -> replay -> pilot, not an automatically launched full matrix. GPU correctness, checkpoint availability, and end-to-end results must be reported as not executed until they are actually run on the cluster.

Official execution references checked on 2026-09-25:
- Slurm job arrays and concurrency: https://slurm.schedmd.com/job_array.html
- Slurm allocated GPU visibility: https://slurm.schedmd.com/gres.html
- Structured output schema behavior: https://developers.openai.com/api/docs/guides/structured-outputs
These support execution/API details only; the scientific design above is a proposal derived from the manuscript and code audit.
