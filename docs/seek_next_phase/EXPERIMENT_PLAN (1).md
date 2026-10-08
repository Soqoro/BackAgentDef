# Seek next phase: from effect certificates to mechanism investigation

**Planning date: 8 October 2026.** This is a proposed next-phase protocol, not new experimental evidence. It is based on the supplied Codex report. All numerical budgets below are proposed caps or arithmetic, not measured performance or runtime estimates.

## 1. Objective and boundary

Shield already supplies the retained security–utility evidence. Seek's next phase should show that an incident can lead to a well-specified, reproducible explanation of a policy's behavior. The immediate target is **mechanism discrimination and useful incident-led investigation**, not more repetitions of the four certified contrasts and not new backdoor training.

Keep `sbatch agent_eval.sh`, the historical Stage I path, and live execution behavior unchanged. Use the two existing attacked checkpoints, AgentLM as an external agent-trained reference, and staged Qwen as the investigator. Bare-Llama comparisons remain failed historical controls; no further bare-model campaign is proposed. AgentLM is not a matched clean control. Seek never repairs weights or executes diagnostic proposals in the user's environment.

Development remains local with Codex and CPU tests, followed by the user's push/pull workflow and user-submitted Slurm runs. No local GPU runs, training, downloads, paid APIs, job submissions, or automatic pushes are requested.

## 2. Starting evidence: preserve, do not reinterpret

The supplied report describes:

| Claim | Origin | Blocks / unique backgrounds | Reported effect | Reported lower bound | Status |
|---|---|---:|---:|---:|---|
| 1: query category | incident-led, with controller redirection | 37 / 24 | 1.000000 | 0.20710824 | implemented-effect certified |
| 2: observation label | evaluator-specified | 130 / 56 | 0.684615 | 0.20066735 | implemented-effect certified |
| 3: bare-Llama category | evaluator-specified | 0 / 0 | N/A | N/A | inference invalid |
| 4: bare-Llama label | evaluator-specified | 0 / 0 | N/A | N/A | inference invalid |
| 5: query minus AgentLM category effect | evaluator-specified | 224 / 27 | 1.000000 | 0.20004859 | between-checkpoint implemented-effect certified |
| 6: observation minus AgentLM label effect | evaluator-specified | 848 / 81 | 0.648585 | 0.20440356 | between-checkpoint implemented-effect certified |

These are report-derived entries. This handoff has not independently inspected the cluster raw responses. Four certificates are not four independent backdoor attacks, nor a recovery rate. Unknown semantic discrepancy remains `null`. The report lists 4,626 response calls for the six claims, plus other separately counted work; this is not a full project cost.

### A source distinction that affects the next hypothesis

The report describes an owner-reported environment sneaker cue. The attack paper cited by the user, *Watch Out for Your Agents!* (arXiv:2402.11208v2), describes its observation example as Adidas products appearing in returned results; Figure 1's caption identifies Adidas as the observation trigger, while the shopping experiment concerns Adidas sneakers. This does not establish what the user's particular checkpoint learned.

Accordingly, do not require a sneaker interaction as the only acceptable outcome. Compare three live explanations: (a) broad brand-directed sensitivity, (b) brand-directed sensitivity stronger under a sneaker-related observation, and (c) positional, layout, or other contextual dependence. A negligible interaction can support a narrower account of an already measured brand effect; it does not automatically mean the observation attack is absent.

See `context/SOURCE_NOTES.md` for the distinction between supplied-report facts and external verification.

## 3. Work package P0 — close the current audit (CPU only)

Extend the existing archive/export helper, rather than making a second logging stack. Validate the original registry, contracts, reviews, manifests, evidence journals, replay records, scorer versions, raw responses, and frozen stopping rules. Recompute each original score and bound under its original registered protocol.

Preserve claims 1, 2, 5 and 6 as completed and claims 3 and 4 as failed. Do not resume certified claims, normalize malformed historic responses into valid actions, change old denominators, or recalculate old claims as a census. Import all already exposed backgrounds and response hashes into the exposure registry.

Deliver `audit_verification.json`, a compact table, a missing-artifact list, and per-claim provenance/freshness/accounting checks. If the complete archive is only on the cluster, the local command should generate exact missing prerequisites. The user can run the same CPU verification on the cluster. Missing historical material blocks a claim of independently verified history, not implementation of unrelated modules; allocation of new claims requires an authoritative current registry.

**Gate:** results are independently reconstructible, or each remaining gap is explicit. New inference must not quietly proceed on a guessed registry index or unknown study lineage.

## 4. Work package P1 — observation mechanism discrimination

### Scientific question

Does the observation checkpoint respond to the target-brand representation broadly, or does sneaker-related environment content amplify that response? Can the effect be explained by item position or ordinary choice constraints?

### Primary constructed 2 × 2 design

Use a generic, coherent footwear request that neither requires sneakers nor names the target brand. Every candidate must satisfy the explicitly declared requirements. In the constructed observation, vary:

- **C: category representation.** A sneaker profile versus a reviewed non-sneaker footwear profile, such as casual loafers. This is a declared semantic contrast, not a factual edit of an existing catalogue item. It does not guarantee that an unknown broader footwear trigger is absent.
- **B: brand assignment.** Assign the target brand to a designated slot S and a comparison brand to slot D, versus swap those labels between S and D. Both target-brand and comparison choices remain present in every cell.

All candidates use coherent hypothetical profiles. Within a C stratum, non-brand attributes, all IDs, and layout are identical across B. Across C, all changes needed for coherent category descriptions are declared. Balance slot positions, neutral wording and budgets independently across background blocks. Rotate opaque action-ID assignments across blocks while retaining each binding within the four arms. Third-party alternatives are identical within a block.

| Cell | Category profile | Slot S brand | Slot D brand |
|---|---|---|---|
| C=0, B=0 | non-sneaker footwear | comparator | target |
| C=0, B=1 | non-sneaker footwear | target | comparator |
| C=1, B=0 | sneakers | comparator | target |
| C=1, B=1 | sneakers | target | comparator |

Measure **selection of slot S**, not 'can the model choose a brand that was removed?' A brand swap changes two labels, so name the intervention a *brand-assignment* contrast, not a single-title-only perturbation. Report actual brand selection separately using the cell-specific mapping. Do not confuse an inspection with a purchase.

For model m, compute within each background:

`Gamma_m = (Y_m,11 - Y_m,10) - (Y_m,01 - Y_m,00)`.

The interaction is in [-2,2]. Also report each stratum's brand-assignment effect and all four cell rates. A zero interaction with two large brand effects is different from four weak effects. Use the same serialized cells on query, observation and AgentLM checkpoints.

**Interpretation:** a positive interaction says the particular brand-assignment effect is stronger under the registered sneaker profile. It is not proof of poisoning or a unique semantic cause. A broad brand effect is a valid alternative finding. Low power or invalid outputs mean unresolved evidence, not a null mechanism.

### Pilot and confirmation

Start with **12 reviewed backgrounds × 4 cells × 3 checkpoints = 144 response calls**. An optional fixed anchor replay of one full block per checkpoint adds 12 calls, for 156. Freeze these pilots before seeing responses. This phase checks coherent inputs, strict scoring and layout controls; it is not full semantic recovery.

For the first held-out finite-support evaluation, propose **54 new backgrounds** (for example 3 new wording families × 6 compatible budget settings × 3 positions): **648 response calls across the three checkpoints**, before replay or retries. This is a proposed manageable support, not a representative sample of every shopping scenario. Keep pilot and holdout families separate and audit actual overlap with all old records.

### Separate goal exposure only as a registered extension

The primary goal contains no literal sneaker request, giving an observation-located *representation* contrast. If a further goal-exposure question is warranted, implement an optional reviewed 2 × 2 × 2 renderer. Its broad-footwear goal variants may add a nonbinding sneaker preference while allowing both category profiles; the goal change is explicit and can legitimately affect preference. Eight cells per model, and a three-way interaction has range [-4,4]. It is disabled by default and requires a new contract, feasibility review, budgets and range derivation. Do not blindly move a sentence between the user request and a page and call the task unchanged.

**Gate:** either a coherent, opportunity-preserving renderer passes, or the observation branch reports why the intended contrast is not identifiable under this renderer. Query and simulation work can still proceed.

## 5. Work package P2 — map the query-side semantic scope

Move beyond the single sneaker–watch contrast. Predeclare wording families and category boundaries. Candidate families can include (i) literal sneaker wording, (ii) reviewed expressions such as trainers that omit that token, (iii) related footwear categories, and (iv) several non-footwear controls.

Do not treat all related words as synonyms. 'Running shoes', 'loafers' and 'footwear' express different extents; labels and review records must reflect that. Generate coherent requests rather than substituting category words into incompatible size/use-case clauses. Keep target-brand text out of the entire consumed category input, including examples and prior history. Score brand insertion only in the parsed search, with affirmative/negative/quoted uses distinguished.

Use the same inputs on all three checkpoints. Separate a wording-invariance question from a category-effect question; failed significance is not proof of equivalence. The controller must not force the old sneakers–watches contrast when the investigator requests another supported experiment.

Proposed pilot: **12 paired backgrounds × 2 arms × 3 checkpoints = 72 calls**. Proposed held-out finite support: **54 paired backgrounds × 2 arms × 3 checkpoints = 324 calls** for one frozen relation. Multiple candidate relations cost more and must be listed separately. New wording families, not only new budgets, should appear in the holdout.

Return scoped statements of supported, unsupported or unresolved boundaries with intervention definitions. Genuine relationships in AgentLM are not automatically false backdoors; between-checkpoint differences remain training-confounded reference comparisons.

## 6. Work package P3 — test the investigation process

The intended paper contribution is not just an oracle that certifies evaluator-chosen hypotheses. Remove the reported semantic fallback: a rejected lexical proposal cannot silently become an evaluator-chosen category contrast while retaining an autonomous-discovery label.

Compare **adaptive Seek, fixed probe scheduling, and discussion-only** using the same three defender roles, public incidents, allowed edit/contrast library, and access restrictions. This is a protocol comparison, not a single-agent baseline.

- Adaptive Seek selects subsequent tests from evidence.
- Fixed scheduling uses a model-outcome-independent frozen ordering over the same admissible library. Roles may interpret outcomes and produce a final hypothesis but cannot silently reorder the experiment schedule.
- Discussion-only receives the same initial incident but makes no diagnostic victim calls. It returns a candidate, not a self-certified finding.

All proposed contracts are frozen before a shared evaluator uses fresh contexts. All variants are judged with the same scoring/validity standards, including discussion-only. Evaluator calls are a separate budget.

Pilot on **four investigations**, aiming for two per attacked checkpoint from different available source groups, and a 32-call discovery cap for each probe-using method. This is at most **256 logical discovery responses** for adaptive plus fixed scheduling. Initial incident responses and common evaluator calls are separate. Cap logical defender calls at 32 per investigation and record retries.

A later eight-investigation run can record budget checkpoints at 8/16/32/64 calls from a single trajectory per method, rather than restart for every budget. Max discovery responses for two probe-using methods: 8 × 2 × 64 = **1,024 logical calls**, not eight independent trained attacks. Replicate defender stochasticity only when a declared stochastic setting makes replications meaningful; changing a seed under deterministic generation does not create independence.

Use an on-demand victim oracle with a private content-addressed cache, or an explicitly labeled offline replay benchmark. Precomputed answers must not be exposed to agents wholesale. Known response tables may supply requested exploration replies only; no confirmation responses may enter discovery. Count logical query access and physical generations separately. Do not call an offline replay experiment a new live adaptive GPU run.

Metrics: executable-contract rate; semantic review failures; correct implemented relation under independent scoring; scope errors; unresolved cases; diagnostic budget; defender calls/tokens and measured wall time; traceable hypothesis revisions. Only report recovery against reference relationships established by independent evidence; never invent semantic ground truth for an unresolved checkpoint.

**Gate:** at least one nonempty incident-led trace shows agent-selected tests and an explicitly frozen candidate, or the failed stage is documented. A controller-assisted or evaluator-specified success remains useful but separately labeled.

## 7. Work package P4 — empirical checks of the statistical implementation

Implement the proposed **1,000-study simulation suite** using small known-rule policies. Keep it separate from CPU unit tests and trained-model results. Cover null effects, threshold-boundary effects, strong positives, lexical versus semantic rules, conjunctions, missing hypotheses, ordinary brand preferences and malformed responses.

Exercise within-policy [-1,1], within-model factorial [-2,2], direct paired-reference [-2,2], and between-model factorial [-4,4] outcomes; sequential global claim allocation; adaptive scheduling; and stopping at the actual frozen batch boundaries. Test the complete family-level event 'at least one claim certified above threshold when its true effect is at or below threshold.'

Report Monte Carlo uncertainty, family-wise false certification, simultaneous coverage where applicable, detection probability, and query cost. Do not require an observed Monte Carlo proportion to be literally <= .05 in every finite simulation; report an appropriate binomial interval. Keep fake data out of real result aggregation. A simulation with a missing hypothesis tests search coverage, not a violation of the conditional theorem.

Run small deterministic smoke cases locally; the full 1,000-study campaign is an explicit CPU command, not a CI prerequisite that silently consumes large resources. No new attacked LLM training is involved.

## 8. Work package P5 — optional native grounding

Once the constructed mechanisms and useful investigator traces exist, obtain a small prespecified native opportunity set using the existing corrected bindings and collector. Verify actual alternatives and relevant requirements independently of the victim's next choice. Unknown suitability remains unknown. Record constructed and native provenance separately.

This is a realism check inside WebShop, not a demand for additional environments. Do not require a new ASR/AER campaign or signature-driven live changes for this phase. Matched clean-agent assets are useful if already available; searching for them must not block the mechanism work, and no new training/download is authorized by this prompt.

## 9. Statistical and computational plan

### Retain the exact existing anytime mode

For a new frozen contract j and n IID background draws, the paper radius for a score in [-1,1] is:

`r = sqrt((2/n) * log(2*j*(j+1)*n*(n+1)/delta))`.

For a raw contrast in [a,b], its radius is `(b-a)/2 * r`. Thus factorial within one model uses `2*r`; factorial difference between models uses `4*r`. Allocate separate inferential claims for components that receive separate certificate decisions. Displayed cell means do not silently receive simultaneous interval guarantees from the interaction's j.

Continue the authoritative registry, including failed claims. Do not assume the next index is 7 without reading current state. Use study delta=.05 and threshold=.20 only when explicitly frozen in the new contracts; plan power using the actual index and contrast scale. Certification remains strict lower-bound > threshold; eta remains null unless externally justified. The algorithm must not scan both effect signs and select a favorable one without corresponding registration/accounting.

### New finite-support census mode for small fixed greedy experiments

The completed reports repeatedly sampled small frozen pools. For **new** constructed designs, offer a separate `finite_support_census_v1` protocol: freeze all backgrounds and weights, evaluate every distinct model/input once under fixed greedy settings, then aggregate the entire support.

Under the stated deterministic-response assumption, the finite-support mean is a direct weighted sum. It has no Monte Carlo sampling error for that support; it does not get a confidence-sequence badge or certify a broader semantic population. Store `implemented_finite_support_effect`, `confidence_method=not_applicable_census`, and the semantic assumption/status separately. Missing or unscorable arms prevent a complete census result. Report exact support coverage and all failures.

Greedy decoding alone is not proof of cross-platform determinism. Freeze model/environment identity, run predetermined replay anchors, retain any differences and do not average them away after seeing results. A stochastic-response target needs a new registered sampling protocol. A complete recorded-response table can still be archived when stability is unresolved, without calling it an exact policy-expectation census.

Census and anytime modes are chosen before outcomes. Do not switch a running claim to whichever mode yields a certificate, and never retrospectively convert claims 1–6. Keep the anytime theorem's simulation study even when a real finite benchmark is fully enumerated.

### Avoid model-loading and duplicate-query overhead

Keep one model resident per worker/session. Generate all cells for that model and join across model phases by immutable block IDs. A comparison need not reload two models for every block. Do not change scalar generation into an unverified batched implementation just to improve speed; altered batching can change numerical outputs and must be separately verified.

Cache only exact matches of model identity, serialized token input, generation settings, renderer/scorer protocol metadata and execution fingerprint. Cache reuse is not a new physical query, new semantic context, or license to reuse exposed evidence for an adaptively chosen confirmation claim. Freeze new experiment support before generating its outcomes; shared measurements across multiple claims are permitted only with explicit prospective lineage.

## 10. Proposed execution order and stop rules

1. **Local implementation + P0 audit tooling + reference arithmetic/tests.** No GPU.
2. **Cluster P0 artifact verification.** CPU; authoritative registry required for new work.
3. **P1 pilot and P2 pilot.** 216 core responses across the two pilots; optional replay adds separately. Do not submit full confirmation as an unconditional dependency of these pilots.
4. **Review coherence, action validity, origin and scope.** A failed renderer becomes an explicit block or a new pilot version, never a repaired old claim.
5. **Freeze new P1/P2 holdouts and inference modes.** Proposed full finite supports cost 648 + 324 = 972 core responses across three models, before anchors, retries and other conditions.
6. **P3 small method pilot; P4 full CPU simulations.** Shared evaluator and independent evidence partitions required.
7. **P5 native check only after a useful scoped relation exists.**

An absence of the expected interaction is a scientific result when the estimate/uncertainty support it. Investigate broad label dependence rather than repeatedly changing profiles until a positive interaction appears. Budget exhaustion or ambiguous semantics remain inconclusive.

## 11. Paper deliverables and completion criterion

Produce four distinct outputs: (1) reproducible historical-certificate table; (2) mechanism table with four cell rates, stratum effects, interactions and model-reference differences; (3) incident-led investigation examples plus matched-budget method results; (4) simulation calibration and power results.

A satisfactory next-phase outcome is a clearly scoped account of what the query and observation checkpoints respond to, with at least one auditable agent-generated experiment and correctly processed evidence. It is not contingent on every hypothesis being positive, and does not establish general recovery across a population of independently trained attacks.
