# Revision notes

## Narrative and structure

The introduction now moves from hidden trigger-dependent behavior, to the additional observation route and external consequences of agents, to the need for useful continuation, and then to complementary multi-agent containment and diagnosis. Trigger form (lexical/semantic) is distinct from source channel (query/observation). The black-and-white attack-definition box is retained and generalized beyond shopping actions.

Problem formulation is folded into Methodology; setup, both result stages, ablations, qualitative analysis, and limitations are folded into Experiments. There are five main sections. The name remains Shield-and-Seek.

## General formulation

The method acts on user intent, grounded state, structured proposals, executable input conditions, and measurable behavior. Domain adapters provide concrete fields and renderers. WebShop is motivated by its ability to expose the complete input-condition → proposed action → executed action → task reward chain. It is a controlled testbed, not the definition of the method or evidence of universal transfer.

A category-level or equivalence-class signature is a valid diagnostic target when supported within its scope. The text no longer requires uniquely reconstructing an internal backdoor mechanism. An unrequested output target is separated from an explicit user prohibition. Raw and consumed observations are distinguished: a cue removed by Shield before inference is not treated as an exposure for that call.

## Algorithmic Seek

HACS is now integrated into the method rather than left as a possible future improvement. Hypotheses are response rules, not merely names of contrasts. The controller maintains a bounded, executable family, chooses comparisons by arm-wise prediction disagreement per logical response cost, and provides periodic coverage. It updates hypotheses from both outcome arms, retaining the difference as a separate quantity. High/high and low/low zero differences are not collapsed.

The Goal, State, and Action roles review semantic meaning, executable comparisons, outcomes, and remaining alternatives. Unknown predictions are unassessed. Tied hypotheses remain grouped; repeated tests require an evidence-gathering purpose. Development-informed vocabulary and evaluator-specified contrasts retain their origins. The 0.1/0.9-style search interpretation is not presented as a calibrated posterior.

## Theory

The two existing results—adaptive soundness and conditional detectability—are retained with compact main-paper statements and full appendix proofs. The revised assumptions explicitly bind the renderer, scorer, context distribution, generation procedure, and prospective evidence. Unknown semantic discrepancy is never set to zero. A measured implemented effect is distinct from a stronger semantic claim. The text does not claim a logarithmic or universal recovery theorem for the HACS heuristic.

The non-interference statement includes isolation of state, randomness, timing constraints, and action budgets. Between-policy contrasts use their wider bounded range. Signed differences and equivalence claims are treated separately.

## Shield data

All 19 original rows from the two baseline tables, component ablation, and cross-category table are retained verbatim, including the reported means and original bold choices. The primary model/output and runtime tables each add lexical-query/observation ASR and AER placeholders plus a clean-input utility placeholder for the new policy. No existing value is copied into a lexical cell.

The old comparison labeled “Lexical guard” remains a defense baseline, not a lexical attack condition. Preliminary OS/Mind2Web figures from the supplied appendix are preserved as preliminary Shield evidence.

## Seek evaluation

The new main table compares Clean, Lexical, and Semantic policy conditions in the query channel. Functional recovery is primary; exact marker recovery is lexical-only. Investigation coverage, off-condition behavior, clean alarms, inconclusive findings, and victim/defender costs are included. Ordinary signatures on a clean policy are not automatically errors. A separate backdoor-alarm decision must be defined before claiming classifier specificity.

The prospective effect table records signed effects, lower bounds, and excess over a matched clean policy. Statistical false-effect certification is assessed using known-effect simulations, not by assuming every clean model has zero behavioral effects. Role-selected, fixed, and active schedules use common capabilities and independent candidate-specific evaluation.

All new results remain TBD. The qualitative examples are explicitly illustrative. Prior inspected development cases have not been imported as fresh discovery, and the worked examples are not new results.

## Source and build corrections

The broken `Table~` / `ef{tab:model}` reference in the uploaded source was replaced by a valid LaTeX reference. Missing MetaGPT and ChatDev BibTeX entries were supplied; references for active search and time-uniform inference were added. BadAgent and CROW bibliographic metadata were updated from primary publication pages. The Watch Out for Your Agents citation now explicitly anchors the query/observation attack construction, avoiding conflation with the separately authored BadAgent paper.

ReAgent is absent from the manuscript and the used bibliography. No claim is made that every listed work was exhaustively re-audited, that the official AAAI 2027 format was validated, or that the expanded proposal meets a submission page limit.
