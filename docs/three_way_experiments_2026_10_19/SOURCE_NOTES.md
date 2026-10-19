# Evidence and implementation sources

## Basis

This handoff uses the revised proposal, the 18 October report, the uploaded v4 execution-review source ZIP, and an older full repository ZIP. It does not claim inspection of the live GitHub checkout or current cluster filesystem. `context/SOURCE_FINGERPRINTS.json` records the uploaded inputs actually inspected.

### 18 October report (provided by user)

`context/CODEX_REPORT_2026-10-18.md` is a preserved copy of `Pasted markdown(5).md`. It reports:
- Completed two-template $150 query comparison: no Adidas for boots, Adidas for athletic shoes, paired difference −1; four probe plus two anchor responses, stable repeats.
- This is development evidence, not a new scientific certificate or fresh method study.
- HACS, source/serialization guards, finite-support and simulation interfaces exist.
- Earlier reasoning review did not pass; method matrix remains locked.
- No completed matched clean/lexical/semantic campaign, large simulation study or resolved observation mechanism.

These are report-derived facts, not independently rerun model results. The new campaign does not inherit the old execution approval.

### Revised manuscript (provided/generated in this conversation)

`Shield_and_Seek_Revised_Proposal.tex` and `context/PROPOSAL_EXCERPTS.md` specify:
- Generic action-boundary formulation and HACS discovery.
- Retained Shield rows, lexical-query/observation placeholders, and a query-channel Clean–Lexical–Semantic comparison.
- Functional/semantic-region recovery instead of mandatory exact mechanism recovery.
- Separation of statistical effect claims, unknown semantic discrepancy and attack attribution.
- Joint ASR/AER evaluation and explicit control matching.

The manuscript's proposed experiments are not already measured results.

### Uploaded execution-review source snapshot

Selected directly inspected paths inside `seek_query_v4_execution_review.zip`:
- `seek/discovery_phase/hacs.py`: word-pair capability, grammar-generated predicates, prediction vectors, active/fixed/role-selected search, unknown/equivalence handling.
- `seek/next_phase/design.py`: a finite product-word library and constructed category/brand experiments.
- `seek/lexical_diagnostic.py`: a historical sneakers/sneaker-shoes/trainers lexical substitution diagnostic, not arbitrary-marker discovery.
- `seek/discovery_phase/hacs_cli.py`, `methods.py`, `campaign.py`: reuse candidates for new campaign orchestration.
- `WebShop/test.py`: original attack-event and reward implementation must remain preserved.

The archived evaluator explicitly distinguishes main trajectory ASR from strict purchase ASR. Its query event is target addition in first search plus a real target click; observation event is target click after the registered observation condition, excluding a query-like first search. The new lexical marker requires a separately versioned exposure detector. These source details must be rechecked against the current checkout, not assumed unchanged.

### Older full repository snapshot

`BackAgentDef-main.zip` contains training entry points and `WebShop/create.py`. The archived `train_fastchat.sh` has a four-GPU full-tuning recipe and an observation `poison_m50.json` path. `train_lora.sh` is a different recipe. This does not bind any current checkpoint to either run.

The archived `create.py` has teacher steering and explicit query-marker persistence, but its legacy observation procedure favors target-brand products rather than providing the newly proposed independent lexical-observation marker. Reuse validated simulator/serialization pieces, not the claim that the new attack already exists.

## External primary sources checked for this handoff

1. Yang et al., *Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents*, arXiv:2402.11208v2 (2024), Sections 4.1–4.3. The paper describes separate query/observation attacks and both a clean model and clean extra-trace controls; the channel-specific extra-trace clean models are not automatically identical. This supports deliberate control matching, not an assumption about the user's checkpoint provenance.
   Source: `https://arxiv.org/html/2402.11208v2`
2. Official Slurm job-array documentation. Array concurrency can be capped; the runbook should use explicit, budgeted submission rather than hidden launches.
   Source: `https://slurm.schedmd.com/job_array.html`
3. Official Slurm GRES documentation. GPU visibility is supplied by Slurm and must not be replaced with a physical device ID from an old machine.
   Source: `https://slurm.schedmd.com/gres.html`
4. Official PyTorch 2.10 reproducibility notes. Reproducibility is not guaranteed across all platforms/releases; freeze actual runtime settings and use explicit checks instead of equating greedy generation with universal determinism.
   Source: `https://docs.pytorch.org/docs/2.10/notes/randomness.html`

These sources inform the plan. They do not change the user's scope to prompt injection, new environments, or internal-model forensics. No public research source substitutes for the private training/run manifests.
