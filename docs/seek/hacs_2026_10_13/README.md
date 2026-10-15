# HACS implementation delivery — 13 October 2026

Subsequent cluster failure and correction: [14 October outcome compatibility fix](OUTCOME_COMPATIBILITY_FIX.md).
Later abstention review: [allocation 14 grounding interface correction](GROUNDING_TYPES_FIX.md).
[Allocation 15 grounding inventory and bounded development diagnostic](GROUNDING_DEVELOPMENT.md)
supersedes the recommendation to collect further fresh incidents while debugging grounding.
The delivery results below describe the original version; the correction requires
a new source-bound pilot and tokenizer check.

Implemented as an optional policy inside the existing Seek discovery entry point. No local GPU/model calls, downloads, training, paid API calls, Slurm submissions or git push occurred.

- [Exact gated cluster commands](NEXT_ACTIONS.md)
- [Delta audit](HACS_DELTA_AUDIT.md)
- [Algorithm and compatibility/limitations](ALGORITHM.md)
- [Capability coverage](CAPABILITY_COVERAGE.md)
- [Review/report template](REPORT_TEMPLATE.md)
- [Validation receipt](validation/receipt.json)
- [Source manifest](validation/execution_source_manifest.json)
- [Simulated arm-update demonstration](validation/simulated/delivery_demo.json)
- [Two-study simulated smoke](validation/simulated/delivery_v1/result.json)

The implementation derives a typed hypothesis family from a declared domain graph, records literal public-input spans, grounds the outcome in the observed action, commits arm predictions before responses, updates exploratory Brier weights once per new consumed-input/runtime fact, and chooses separating tests with disagreement/cost plus every-fourth coverage. Fixed and role-selected policies share the same capabilities, reducer and guards. Repeated requests receive an explicit decision; no evaluator-chosen contrast silently replaces them.

Compact Action/Goal/State packets include graph provenance, grounded features, recent exact facts, predictions/losses, actual last inputs/actions and unperformed tests. Numeric acknowledgments are checked, but qualitative reasoning and semantic scope still require independent review. New evidence and source manifests are immutable; global allocations/exposures reuse the authoritative registry. Old Shield behavior, claims, failures, results and prior review gates remain unchanged.

The common finite-support evaluator reserves all primary contracts together, retains all method outcomes in denominators and privately shares identical physical responses. Evaluation never becomes feedback to search or a confidence certificate. Existing confirmation arithmetic and strict thresholds are untouched.

Validation: **311 Seek CPU tests**, including 34 HACS tests and one new opt-in single-group collection test; **85 selected Stage I tests**; **36 agent_eval.sh dry-run rows**; **37 supplied reference tests**. All **24 protected hashes** match before/after. Tests use fake policies or known-rule simulations, not real model verification.

The final two-study smoke used 15,480 simulated logical draws, observed zero family false certificates with interval [0, 0.6576197725], and four certificates among 28 positive cases. Forty-four confirmation streams reached their 64-block cap without certification. This is software smoke evidence, not demonstrated calibration, power or superiority. The full 1,000-study campaigns remain unrun.

Known limits: unknown query-adapter predictions remain for position/ID/task/source/history hypotheses; domain relations are development-informed priors; agent ontology edits need a new reviewed version; no ordinary exposure ledger is assumed; interrupted runs require review rather than automatic replay; new prospective real inputs and actual tokenizer/model compatibility remain cluster prerequisites. No real HACS pilot, held-out outcome or semantic recovery is claimed. The first cluster step is to freeze the library and acquire **one new** eligible group; the later matrix needs its own independent gate and four other fresh groups.
