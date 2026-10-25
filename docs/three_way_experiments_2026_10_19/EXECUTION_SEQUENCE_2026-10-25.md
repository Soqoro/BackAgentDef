# Experiment sequence agreed on 25 October 2026

This records the user's requested execution order. It reorganizes upcoming work;
it does not approve resources, training or inference, change frozen experiments,
or erase historical failures. Existing preparation components are reused.

## 1. Construct lexical training examples

Prepare reviewed task-grounded query and observation source slots with separate
private literal markers. Use active, absent, decoy and near-match inputs. Give C,
LQ and LO the same input exposure; change the intended next-action target only
for the corresponding poisoned arm's active-marker state. Preserve benign
context and apply the reviewed annotation-aware supervision policy consistently.

Deliver final corpora, sidecars, target-action bindings, marker-token receipts,
source/split manifests and native mask receipts. The 1,858-record candidate and
CPU adapter are preparation assets, not already cleared final training corpora.
For newly recorded tasks, capture task, product, trajectory and template lineage
prospectively. Unresolved historical lineage remains explicitly unresolved.

## 2. Train matched clean and poisoned agents

Train C, LQ and LO from the same verified initialization, with the same resolved
recipe, source inputs, data ordering and training budget. Preserve initialization,
trainer, tokenizer, environment, optimizer, seed and actual run receipts. Different
labels can produce different token counts; record rather than conceal them.

Do not initialize the primary lexical-only arms from the old attacked checkpoints.
Any such continued-fine-tuning experiment is a separate mixed-history condition.
Install/review the opt-in launch gate and obtain scoped execution approval first.
Multiple seeds, if run, are distinct training replicates; evaluation episodes are
not training replicates.

## 3. Verify that the trigger works

Freeze readiness criteria and benign-utility floor before inspecting results.
Check active versus absent/decoy/near-match states, matched C behavior, action
validity, truncation, input serialization, target opportunity and anchor stability.
Verify query and observation channels separately. Observation delivery must
follow a clean prefix, with raw and actually consumed cue exposure distinguished.

Next-action readiness is not purchase-level ASR. Use the bounded episode pilot
for episode functionality and utility. Failed attacks remain reported failures;
record all development variants and do not tune on final evaluation tasks.

## 4. Evaluate Shield: ASR and AER

Run undefended versus Shield pilots, then the frozen evaluation panel after pilot
review. Report query and observation results separately. Use the same assigned
tasks for paired comparisons and record failures and missing episodes.

ASR follows the explicitly versioned attack-event definition and denominator;
retain strict purchase success separately. AER is average episode reward, not
probe deviation or trigger-detection rate. Include benign-input utility for each
checkpoint. Preserve raw exposure, consumed exposure, proposed action, executed
action and purchase fields. Do not condition the primary denominator solely on
post-Shield trigger survival. Historical semantic rows retain their old definitions.

## 5. Evaluate Seek: Clean versus Lexical versus Semantic

Use common public capabilities, response budgets, investigator interfaces and
prospectively assigned evaluation tasks. Keep private trigger labels out of role
packets. Preserve actual agent candidates and label controller choices as
algorithmic. Count no-incident, abstention, invalid and failed investigations.

The existing primary query comparison is C/LQ/SQ. An observation comparison
C/LO/SO requires its own reviewed panel rather than pooling channels. Legacy SQ/SO
are descriptive unmatched comparators unless new matched semantic replicas are
trained and verified. Their inclusion cannot support a causal lexical-versus-
semantic training comparison by itself.

Report functional scoped recovery, exact marker recovery separately, coverage,
false alarms under a declared rule, cost and unresolved alternatives. A genuine
clean-policy dependency is not automatically a false backdoor certificate. Keep
finite-support findings, statistical certificates and semantic interpretation separate.

## 6. Run HACS ablations and finalize tables

Execute the preregistered ablations after the preceding evaluations. Freeze the
ablation choices, metrics and budgets before inspecting final test results; the
execution order does not permit retrospective test-set selection. The existing
primary contrast is active versus fixed HACS with matched capabilities and budgets;
additional ablations remain separately specified and reviewed.

Finalize tables with model identities, matching status, seeds, task counts,
denominators, uncertainty, failures/abstentions, costs and scope. Keep training
replicates distinct from repeated investigations and episodes. Retain unsuccessful
runs and review verdicts.

## Current position

We are in stage 1. Corpus conversion, supervision-policy review and benign adapter
CPU validation have progressed; final lexical examples/corpora, launch gating and
verified provenance/splits remain incomplete. Historical semantic runs and bounded
HACS development results do not count as completion of stages 2–6 for this new
campaign. All new model-execution approvals remain false.

Use the existing [cluster runbook](CLUSTER_RUNBOOK.md) for component commands,
subject to this order and current prerequisites. The
[24 October audit report](../seek/reports/SEEK_PROGRESS_AND_AUDIT_2026-10-24.md)
preserves the detailed evidence and outstanding limitations.
