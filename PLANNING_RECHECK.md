# Planning recheck and corrections

**Date:** 28 October 2026. This reviews the previous planning documents, not the current repository implementation or private balanced-packet bytes.

## Sources examined

- `Seek_LQ_Training_Readiness_2026-10-28/CODEX_PROMPT.md`, raw SHA-256 `1ef8e6182fb41b454d11ee48f4e54571f41641580dae3ff0528d181715b0824f`.
- `Seek_LQ_Training_Readiness_2026-10-28/EXPERIMENT_PLAN.md`, raw SHA-256 `e1c12d31a13682c907573943eb98f940165c94577c9345e8657aadf831e18746`.
- The user's pasted balanced-development review, including its explicit limitations. The underlying private marker/mapping was not inspected for this planning revision.
- Official PyTorch CrossEntropyLoss documentation was consulted for the distinction between ignored target labels and inputs and for nonignored-target averaging; this is general technical context, not verification of the installed FastChat loss implementation.

## Overall judgment

The C/LQ-first direction, brand-agnostic target, next-action additions, immutable history and separate execution gates are retained. The earlier plan contains overrestrictive criteria and several ambiguities that could create misleading results or further stall the work. The v2 files replace those planning instructions without altering historical evidence.

## Findings and repairs

| ID | Previous issue | Correction in v2 |
|---|---|---|
| P01 | The marker-exposed, benign-label C arm could be described as a general clean baseline. | Name its interpretation C_sham; retain C as a code alias. It is clean relative to the newly planted association but deliberately exposed to the cue with benign targets. Its false-alarm results do not characterize all clean agents. A generic clean-reference study is optional follow-on, not a new blocker. |
| P02 | Requiring <=4/16 target outcomes in every C/control cell excludes genuine strong conditional effects with substantial natural background behavior. | Use paired active-minus-control effects and excess over sham. Proposed relative thresholds are explicit engineering settings, not statistical guarantees. Preserve raw arm rates and separate validity, activation, attribution and utility statuses. |
| P03 | 'Preserve every original annotation' was ambiguous for additional next-action examples whose intended objective is final-action-only. | Preserve original annotations on base records. New additions have their own explicit context-only prefix and final-action target annotations, with original annotations separately preserved. Exclude original future responses/observations from the prefix. |
| P04 | A changed search was insufficiently connected to the desired Shield security endpoint. | First characterize Y as conflicting, goal-compatible or only an output transformation. Seek can measure any supported signature; Shield claims need the registered security endpoint. Do not require same-page products for a search-only target or declare every altered search forbidden. |
| P05 | 'Strongest' product/template grouping could be interpreted as requiring universal disjointness, including shared standard prompts. | Name the held-out axis. Prevent task/trajectory leakage; disclose shared generic interfaces and assess specific demonstration dependencies. Product/template disjointness is an extra claim, not silently mandatory. |
| P06 | A new unused readiness pool was required after every training revision, although readiness is explicitly development. | Permit bounded, recorded development reuse. Do not count reuse as independent evidence. Protect the final evaluation from outcome-driven selection; additional model runs still need separate grants. |
| P07 | Large downstream populations and universal two-category diversity could become prerequisites for the first candidate. | Fifty training tasks remains a proposed target, not a fabricated quota. Two categories are desirable only if the fixed mapping applies. Prepare later split infrastructure without making all later benchmarks a prerequisite to coding/training-plan review. |
| P08 | A private mapping could work only as a lookup over inspected IDs. | Require a defensible executable Y(q,s,a) rule on unseen eligible tasks or explicitly restrict the claim. Do not invent a replacement Y. |
| P09 | Decoy/near controls included during training were liable to be presented as unseen lexical-specificity tests. | Label them seen negative cue strings on new tasks. A later unseen-string/placement challenge gets a separate frozen budget; the 132-call pilot is unchanged. |
| P10 | Trainer integration could pass if a later collator silently overwrote the adapter labels. | Add an explicit label-preservation invariant and a deliberately mask-overwriting negative fixture through the real CPU entry-point path. |
| P11 | Method comparison lacked an explicit prohibition on sharing adaptive histories and cached sampled outcomes. | Isolate methods; charge logical probes even with valid deterministic cache reuse. Do not share stochastic samples as fresh calls. |
| P12 | A single generic readiness decision obscured measurement failure, effect absence, attribution uncertainty and task incapacity. | Use separate statuses and freeze absolute utility plus paired degradation criteria before execution. Equal bad control/poisoned performance is not usable. |

## What did not need changing

- Lexical backdoors do not require Adidas. Neither the old fragrance fixture nor a color example determines the new private target.
- Next-action additions can be an intentional initial training design. Full trajectories are not a logical prerequisite, but model learning and full-episode utility remain empirical questions.
- All 350 retained WebShop base records are training exposure; exclusion only from added poison records cannot make them held out.
- Balanced token lengths alone do not establish semantic equivalence or model invariance.
- C_sham and LQ can use the same initial weights/recipe; that does not authenticate source cleanliness or prove exact pairing of stochastic training paths.
- Legacy semantic checkpoints remain a separate, unmatched case; there is no clean causal test of trigger form when target/recipe/interface also differ.
- No new execution/training/resource/installation grant is issued.

## Readiness counterexample and revised calculation

For an illustrative 16-task population, suppose C_sham emits Y on 8/16 tasks in every cue state. LQ emits Y on 16/16 active tasks and 8/16 in each control. Then active-minus-control is 0 for C_sham and 0.5 for LQ; excess is 0.5. The previous raw-control criterion rejected it. The revised relative criteria accept this measured separation if the other gates pass. This is a mathematical counterexample to necessity of the old gate, not data or evidence that these rates will occur.

Unknown binary outcomes are represented as [0,1]. Signed effect intervals are propagated by subtracting the opposite endpoint. They are logical missingness bounds, not confidence intervals. Readiness remains selection/development evidence, even when thresholds pass.

## Boundaries of this recheck

No repository source was edited, no model or environment was executed, no private marker chosen, and no balanced-packet approval issued. Any arithmetic tests supplied with this package validate planning examples only. They do not validate the actual scorer, native tokenizer, trainer integration or scientific provenance.
