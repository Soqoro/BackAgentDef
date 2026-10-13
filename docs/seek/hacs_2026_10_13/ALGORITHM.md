# Implemented HACS v1

HACS is an optional policy in `seek_discovery.py`. Legacy Seek execution, Shield, the manuscript certifier and historical artifacts are unchanged. The domain prior was designed after inspecting historical results; it is not a zero-prior-knowledge discovery claim.

## Representation and capability boundary

`discovery_phase/hacs.py` defines versioned feature atoms and rule hypotheses. Rules predict arm responses; experiments are separate objects. The domain JSON pins forms, concepts, provenance and typed edges. `is_a` must be acyclic; separate `expresses_concept` edges are retained but do not create taxonomic implications. Unknown membership returns unknown. The supplied hand-authored graph is a proposed prior requiring prospective review, not an established semantic ontology.

The generic `word_pair` compiler accepts any two distinct declared forms. It does not inherit the old lexical-only sneakers/trainers restriction or silently relabel a hypothesis. The exact same capability card enters every role packet. A comparator is a neutral label, not a comparison operator. Errors include `unknown_operand`, `identical_arms`, `invalid_comparator`, and `ungrounded_outcome`.

Forms are enumerated lexically, all unordered pairs uniformly. Candidate ordering uses constants, surface forms, concepts and ordinary alternatives; it never reads target-policy results. The family cap is 32 initially, 64 after one expansion. Omitted ASTs are logged. Two-atom conjunction expansion starts an explicit epoch with fresh priors; old observations are not retrospectively called predictions. Domain bounds are 16 forms, 64 nodes and 256 edges. No special transition references a historical successful contrast.

The real query adapter constructs new shopping requests, retaining equal budget and search opportunity. It does **not** retain all incident attributes. Category changes are deliberate diagnostic changes and cannot mutate the live goal. Position, fixed-ID, task-attribute, source and history hypotheses have executable ASTs but unknown predictions under this query adapter. They remain visible in coverage. Real observation work continues to use the existing, separately reviewed bridge panel; HACS v1 does not pretend its query adapter manipulates those fields. Unknown adapter branches require a new versioned adapter rather than guessed probabilities. Agent-authored ontology edits are not accepted automatically in this release; a new reviewed domain version and fresh plan are required.

## Grounding and updates

The first Action review supplies an outcome phrase from the actual public action. The existing strict search scorer verifies affirmative insertion and excludes malformed/negated/ambiguous mentions. Provenance records the exact action span. This is behavioral evidence, not a claim that the user forbids the brand or that the policy was maliciously trained. No target brand is embedded in the controller or role prompts.

Boolean rules predict 0.9/0.1, fixed before responses. These are exploratory smoothed predictions, not calibrated probabilities. Every experiment selection saves its full pre-response prediction table, scores and origin before victim generation. The reducer retains each arm, strict parsed action, raw text, full input/tokens/runtime, missingness, loss, weights and paired difference.

Weights use the mean Brier loss over valid **new** arms, with learning rate 1, in log space. Prediction-equivalent rules share one initial class prior; duplicate phrasings divide that prior. Low weights are never irreversibly pruned. Unknown predictions are not fitted or scored. Evidence identity is exact consumed prompt + tokens + runtime, independent of task/event/arm IDs. Shared or cached greedy arms update once; malformed responses are missing, never zero. This discovery accounting differs from the historical IID-with-replacement confirmation protocol.

The fit diagnostic reports average prequential loss. After at least two events, loss above 0.35 for every predictable rule flags possible misspecification, allows the single bounded expansion, and then stops unresolved if poor fit persists. This threshold is a development heuristic, not a statistical test. Outside-family possibility always remains explicit.

## Selection and stopping

Active utility is summed arm-wise weighted prediction variance divided by logical response cost. It is **prediction disagreement, not information gain**. Utility is explicitly conditional on the predictable subfamily; the report names excluded unknown rules and never advertises complete coverage. Every fourth choice (or numerically negligible utility) uses the ordered unseen separating-test queue. All hypotheses, including downweighted ones, remain in structural coverage.

`hacs_fixed` uses the same frozen bank order independent of response values. `role_selected` receives the same public library and factual packets, and its requested ID is either executed unchanged or rejected. It is a new capability-matched baseline, not a rewrite of the historical failed baseline.

A repeated role request gets one factual correction. A fresh-context replication is admissible only when it names two actual rules whose predictions differ for that experiment. Exact cached reinspection cannot earn another update. Separate counters cover unjustified requests/executions, fresh replication, cache reinspection, no disagreement, missing capability and budget stops. Lack of a separating test yields `insufficient_experiment_coverage`, not a unique trigger. No final scientific answer is retried to obtain a preferred result.

## Roles, costs and persistence

Goal reviews authorization and diagnostic changes; State reviews graph/rendering provenance; Action grounds outcomes and challenges predictive interpretation. The controller's origin is `systematic_library_search_role_reviewed`; each decision names algorithmic disagreement, algorithmic coverage, fixed order or agent selection.

Packets contain public excerpts, eight ranked rules, available experiments, exact recent numeric facts, unresolved alternatives, and unperformed tests. Full history stays in immutable journals. Numeric acknowledgments are mechanically checked; independent qualitative review is still required. Input cap is min(installed cap,4096); output cap min(installed cap,768). Tokenizer-only preflight runs before **each** actual request. Known oversized input is not retried. Schema/factual errors allow one correction; refusal, transport failure and incomplete generations stop.

Pilot limits: 16 logical victim responses, two predetermined first-pair anchors, eight logical role calls, two attempts/call. Matrix limits: 32 responses plus two anchors and 12 role calls/run. The role-selected baseline may exhaust its discussion allowance before using all victim responses; this is reported, not padded with controller choices. Prefixes at 8/16/32 are saved only when reached.

Qwen and victim run sequentially on one Slurm-visible GPU using existing isolation. Victim stays loaded within a probe batch, not reloaded for each pair. Models are released at review boundaries; latency includes reloads. Precision/templates/generation are inherited unchanged. No batching optimization was introduced.

Completed results resume idempotently. Interrupted runs refuse automatic continuation, preserving accepted turns and unknown physical attempts for review. Failures retain status/counts; CLI pre-load failures emit a separate failure artifact with calls not inferred. Model tokens that are not reported remain null. Discovery allocation uses the existing locked global registry, not a new index counter; selection/execution reserve source and consumed-input exposures.

## Inference boundaries and limitations

A final role-reviewed primary includes unchanged experiment, outcome and actual rules; otherwise the unresolved family is frozen. Fresh 12-pair evaluation requires separate independent review, allocation and complete support. Its census uses the existing finite-support primitive and never emits an anytime certificate. Incomplete support or unstable anchors prevents an exact finite-support claim. Evaluator outcomes do not enter search. The common evaluator reserves all method primaries together and privately shares exact runtime/input cache entries. Logical costs and failed/abstaining methods stay in the denominator. Separately allocated single-candidate evaluators still reject cross-stream overlaps; use the common batch for the method comparison.

The original paper radius, strict threshold and coefficient-derived range scaling remain unchanged. New certification must be separately registered through the existing protocol with fresh support/current j; HACS has no automatic certification launcher. Eta remains null. Claim 10 cannot be resumed.

Ordinary exposures are unavailable initially: class priors are neutral and the packet explicitly disallows prevalence estimates from incident-only selection. The existing collector is reused; no Stage I logging behavior changes.

Known-rule simulations run the production search code with fictional domains, not historical answer overrides. They report class-ranking accuracy and misspecification separately from existing effect-certification simulations. Scripted role choices are not LLM results. A separate idealized elimination function checks true-rule inclusion and balanced splits over all reachable version spaces, with explicit independent bounded-noise sampling. Its guarantee is not assigned to soft-weight HACS.
