# Seek with Hierarchical Active Causal Search (HACS)

Status: proposed upgrade, not an already implemented or experimentally validated method. Retain Shield-and-Seek as the framework name. HACS is an internal search algorithm. This note distinguishes production heuristics, existing confirmation guarantees, and an idealized discovery result.

## 1. A two-level contract

**Execution contract:** the existing frozen user intent and constraints govern live Shield execution.

**Search contract:** a versioned candidate vocabulary, feasible experiment library, outcome, cost/coverage budget and evidence-update policy govern discovery. **Confirmation contract:** the selected relation, renderer, distribution, scoring, uncertainty assumptions and stopping rule are frozen before fresh confirmation.

A search hypothesis is not just “compare A and B.” It is an executable prediction of behavior across several conditions. A contrast is one possible experiment on that hypothesis. Keeping these distinct prevents endless repetition of a contrast that no longer separates explanations.

## 2. Start with the observed effect

The Action role constructs an `OutcomeSpec` from the actual victim action and grounded context. For search, an example is insertion of a particular affirmative brand restriction absent from the request. The brand comes from the observed action, never a private poisoning label. Parse errors, missing mappings and ungrounded targets are separate statuses.

Authorization and diagnosis are separate fields. A brand omitted from a request is not universally forbidden. Whether the action is a Shield violation depends on the existing contract and certifier. Seek can characterize a documented behavior without inventing a stronger user prohibition.

## 3. Candidate graph, not a magic taxonomy

Use a bounded typed graph with provenance:

- surface form/phrase atoms from actual public inputs;
- normalized schema-field atoms, such as category and selected attributes;
- reviewed `is_a` ancestors and siblings from local metadata or a declared domain adapter;
- reviewed `expresses_concept` edges connecting different wordings to a concept;
- source/channel/position/history-window conditions;
- small conjunctions as a separately budgeted expansion;
- non-trigger explanations: constant high/low outcome tendency, ordinary task-choice rule, first-position/fixed-ID rule, and unknown/misspecified family.

Synonyms are not automatically taxonomy children. A category label is not validated semantic truth. Unknown field values remain unknown. Every concept needs an executable matcher and every experiment needs a real compiler/renderer capability. Record candidates excluded by a cap, unsupported operations, and any untested boundaries.

Suggested first implementation: at most 32 initial hypotheses, 64 total after at most one expansion, single-atom rules initially and two-atom rules in a separate condition. These are proposed development settings, not measured facts.

Generate candidates uniformly from generic rules: ancestors, siblings, alternate expressions and distant controls. Never encode a special “trainers failed, choose loafers” transition. The winning historical contrast and reports are hidden from study roles. Vocabulary construction after seeing these reports is development-informed; freeze its provenance and evaluate on new sources. No claim of zero prior researcher knowledge is warranted.

## 4. Feasibility and identifiability before expensive discussion

Compile a finite experiment bank and its hypothesis predictions before victim calls. Make an explicit capability matrix: which hypothesis pairs are separable by which executable comparisons?

Group hypotheses with identical predictions on every available experiment into a **prediction-equivalence class**. A family can be plausible yet unidentifiable using the installed tools. Return `unresolved_equivalence_class` or `insufficient_experiment_coverage`; do not guess a representative and call it a recovered trigger.

This directly addresses the audit's capability restrictions. A semantic graph is not useful when its branches cannot be rendered as coherent inputs.

## 5. Outcome vectors and predictive updates

For a paired test store both values `(Y_1,Y_0)`, the difference, valid/missing counts, raw responses, source/semantic checks, and exact treatment hashes. Record `(1,1)`, `(0,0)` and malformed cases separately.

Before the outcome, every eligible hypothesis h records an arm-wise prediction vector p_h(e). These can be probabilities from a declared exploratory model. For Boolean rules, a smoothed high/low predictor is a practical initialization; its values are heuristics, not calibrated backdoor probabilities. Constant-effect alternatives must remain available. A constant model with no cue dependence is distinct from “no target behavior.”

Use a transparent bounded predictive loss, for example an average arm-wise Brier loss on new valid evidence:

    ell_h(e) = mean_a (p_h(e,a) - Y(e,a))^2
    w'_h ∝ w_h exp(-lambda * ell_h(e))

Scores and predictions are fixed before the response. Accumulate each actual evidence unit once. Shared original arms and cached greedy outputs are not independent new observations. The production score is an exploratory ranking, not a posterior or an error-controlled rejection. Keep low-weight hypotheses available for the coverage policy; do not irreversibly discard a semantic condition on four examples. Unknown predictions go to a coverage queue rather than being silently assigned zero variance.

A deterministic reducer generates the new search state before any LLM is asked to comment. It records what each hypothesis predicted, what was observed, the loss/rank change, unresolved alternatives, and eligible next tests. An LLM cannot alter these arithmetic facts.

## 6. What the latest audit example should do

Idealized illustrative predictions:

| Hypothesis | Sneakers arm | Trainers arm | Nearby non-sneaker footwear arm |
|---|---:|---:|---:|
| Exact observed word is required | 1 | 0 | 0 |
| Sneaker-like concept | 1 | 1 | 0 |
| Footwear broadly | 1 | 1 | 1 |
| General target-brand tendency | 1 | 1 | 1 |
| No target tendency | 0 | 0 | 0 |

The observed `(1,1)` weakens the strict exact-word hypothesis and leaves broader alternatives unresolved. It does not prove an equivalence claim from four pairs. A useful next comparison separates the surviving conditions, possibly through a reviewed sibling or distant category. The scheduler chooses from the actual capability library, not from the literal names in this example.

Changing a category deliberately changes task meaning and is permitted only as a declared diagnostic contrast. Equivalent-wording tests have a different purpose. Neither alters the live user goal.

## 7. Choose tests by disagreement, with a coverage fallback

For complete predictions and unresolved hypotheses, a practical score is:

    U(e) = sum_a Var_{h~w}(p_h(e,a)) / logical_response_cost(e)

Use logical responses, not free cache hits, in the scientific budget. Keep measured physical inference, tokens and latency separate. This score is a disagreement heuristic, not exact information gain. More elaborate acquisition rules are unnecessary for the first patch.

Every fourth selection is reserved for a fair coverage queue of unresolved separating tests/condition families, where such a test is valid and within budget. Initial family weights and queue ordering are frozen and generic. Coverage prevents an observational ranking or mistaken prior from permanently suppressing a useful branch. Unknown/capability-ambiguous branches must be reviewed rather than rewarded as apparently informative.

Retain a recorded counterexample to each proposed broad explanation: name an available condition where its prediction differs from a plausible alternative. Do not require a linguistic “weakener” without an executable test. If none exists, report the indistinguishable scope.

Observation hypotheses include brand following, position following, fixed-ID following and explicit-task-attribute following. Measure each on matched action opportunities, not by removing the only target option. Persistent first-position outputs are data about the tested format, not automatically a software failure or proof that the policy is clean.

## 8. No-progress policy

A repeated request needs a machine-checkable reason:

- new independent background evidence intended to reduce declared uncertainty;
- a preregistered equivalence or effect confirmation step;
- a technical-validity change recorded as a new version and scope;
- a specific unresolved hypothesis pair still distinguished by the comparison.

An exact repeated greedy prompt under the same fixed runtime is a cache/replay check, not new search information. Record it but do not update weights twice. Identical already-resolved comparison requests are not executed again as discovery merely because the LLM repeats them.

After no feasible informative test remains, allow one bounded graph/library expansion with explicit authorship, then return an unresolved class, capability limitation or budget stop. Never substitute a researcher-known successful comparison in secret.

## 9. Compact multi-agent collaboration

Agents still have substantive roles:

- Goal: grounding semantic meaning, separating deliberate factor changes from accidental changes, protecting the live contract.
- State: mapping grounded features and action IDs, proposing or reviewing concept edges and renderable comparisons.
- Action: defining the measurable behavior and challenging the relationship between rule predictions and observed actions.

Use a compact factual packet: incident excerpts, goal/behavior distinction, candidate IDs/rules, current arm counts, pre-test predictions, automatic evidence update, and a small number of feasible next tests. Keep full text in an immutable ledger, not repeatedly concatenated into every context.

Suggested input target ≤4,096 tokens and output cap ≤768 tokens per review, subject to actual backend limits. Preflight exact tokens and reject over-cap input before inference. Review at initialization, after a small probe batch or material family change, and before finalization; do not require three long replies after every victim call. Bound one factual repair; contradictory or invalid review yields a reasoned stop, not a hidden controller approval.

Call the outcome `systematic_library_search_role_reviewed`. Candidate origins include grammar, agent extension and evaluator. Search choices are openly algorithmic. This is not presented as entirely spontaneous LLM discovery.

## 10. Screening evidence and selection bias

The immutable Incident Ledger supplies interesting deviations. A bounded Exposure Ledger can retain ordinary calls or a prospectively specified reservoir sample. Record inclusion probabilities/source cohorts. An incident-only table cannot estimate normal exposure rates.

Contrastive screening can order candidate conditions, but observational frequencies neither prove causality nor justify zero prior weight. A small matrix of all outcomes, including high/high and low/low pairs, is more useful than counting suspicious words alone. Never leak confirmation observations into screening.

## 11. Final confirmation stays separate

Freeze a candidate relation or an equivalence class and its exact comparison procedure. Discovery scores are not effect certificates. Fresh candidate-specific evaluation may produce:

1. A finite-support implemented effect, with complete coverage and no statistical-confidence badge.
2. An anytime implemented-effect certificate under the registered sampling assumptions.
3. A semantic interpretation conditional on a justified discrepancy bound; eta remains null when unknown.
4. Inconclusive, non-identifiable or inadequate-support output.

All claim indices continue the authoritative global registry. No new experiment may reuse historical held-out data as fresh evidence. No edit changes Shield or removes poisoned weights.

## 12. Theory: what can be strengthened responsibly

The production Brier/disagreement strategy is a search heuristic. Keep the existing manuscript's soundness and detectability results for prospective effect confirmation. They do not prove correct semantic candidate construction.

A useful idealized discovery proposition is the following, based on generalized binary-search reasoning rather than a new statistical invention.

### Conditional finite-class identification

Let a frozen finite family H have N≥2 candidate response rules, including a true rule h*. Available valid tests return binary labels predicted by each rule. Predictions must be correct for the defined test distribution; the semantic compiler is not assumed correct merely because an LLM accepts it. Suppose every reachable version space V with more than one distinguishable rule admits a test whose two predicted-label subsets each contain at most (1−beta)|V| rules, for a fixed 0<beta≤1/2. Keep only rules matching the observed label.

Under noiseless answers, at most

    R = ceil(log(N)/[-log(1−beta)])

selected tests suffice to identify h*. Proof: each valid response preserves h* and reduces the surviving set by at least factor (1−beta); after R rounds its size is at most one. Indistinguishable rules can only be identified as an equivalence class. Without the balanced-split assumption, pairwise separability supports a looser N−1-test elimination bound, not logarithmic complexity.

For noise, additionally assume each fresh response to a selected test is wrong with probability at most nu<1/2 and the repetitions satisfy the independence/conditional concentration assumption. Repeat each selected test k times, using a majority vote, where k is odd and

    k ≥ log(R/alpha) / [2(1/2−nu)^2].

Hoeffding bounds an erroneous majority by exp(−2k(1/2−nu)^2)≤alpha/R. Conditional on correct preceding decisions, the same bound holds for the adaptively selected test. A union bound bounds any wrong decision over R tests by alpha. Identification therefore succeeds with probability at least 1−alpha using at most Rk test responses, times the physical arm cost required to implement an oracle test. Cached deterministic repeats are not independent noise samples.

This proposition is for the stated elimination oracle, not automatically for the production soft-weighted search. Implement its assumptions in known-rule simulation to test the matching elimination backend, or keep it as a design note. A hierarchy by itself does not imply balanced tests, inclusion of h*, causal identifiability or correctness of semantic rendering. Do not add an unconditional HACS theorem to the paper.

The practical contribution to test is executable semantic search plus evidence-linked revision and prospective effect confirmation. The theory specifies where search can be systematic rather than lucky, without erasing the hypothesis-coverage assumption.
