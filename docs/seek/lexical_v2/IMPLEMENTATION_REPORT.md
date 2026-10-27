# Lexical v2 implementation report — 2026-10-27

## Result

Implemented an additive **brand-agnostic exclusive-color X→Y preparation path**,
prioritizing LQ. It provides private configurable rules, deterministic source
eligibility and lineage checks, a separate same-state LO audit, matched cue-state
C/LQ compilation, exact-action scoring, annotation-aware masks/prefix validation,
and frozen future milestone drafts. No existing source file was edited by this
implementation; all new code is opt-in. No new search campaign occurred.

**Working local slice:** 4 synthetic tasks across 2 categories, 4 cue states,
2 roles = **32 next-action records**. All labels/masks/prefix/cache/shuffle checks
passed using the explicitly symbolic character-token backend.
**Real eligible sources: 0 supplied locally. Native runtime validation: pending.**
This is not a real-source feasibility certificate, trained backdoor, full-episode
training dataset, attack-effectiveness result or completed Shield/Seek experiment.

## Code and reusable entry points

- `seek/three_way/lexical_v2.py`: private contract validation, source spans,
  query substitution, LO state eligibility, deterministic bounded scanning,
  category-balanced development selection, proposed/executed exact-action scorer,
  public-packet allowlist and matched C/LQ compiler.
- `scripts/seek/prepare_lexical_v2.py`: exclusive private artifacts; source and
  helper checksums; synthetic or real-export mode; optional native-tokenizer-only
  replay; frozen future milestone plans. Never imports a model inference class.
- `configs/seek/lexical_v2/contract.json`: unapproved public rule-family template.
- `tests/seek/test_lexical_v2.py`: 28 new tests including negative cases.

Reuse: `validate_splits`, `marker_manifest`, `action_ok`, `AnnotationDataset`,
`annotation_mask`, `native_processor`, `policy_interface.render`, `prefix_check`,
`freeze`, `private_write`. Existing public-span HACS capability remains available
for future blinded integration; it is not modified or fed private rule literals.

Inspected original `WebShop/create.py` query/observation steering and cue-persistence
checks, `train_fastchat.sh`, legacy `WebShop/test.py`, source notes/proposal excerpts,
old rejected lexical slots, adapter request-changes/approval reviews and current
readiness/metric interfaces. `AER` remains **average episode reward**.
The native FastChat trainer file, local tokenizer, benign source data and root
`REVIEW.md` were absent; no source contents were invented for them.

## Per-slot decisions and counts

The software fixture deliberately uses illustrative color values in test code;
production marker/rule truth is supplied privately, not inherited from the test.

| ID | Category/task | Decision | Reason/limit |
|---|---|---|---|
| fixture-0 | Bags / backpack | Eligible software candidate | One explicit positive color; category and budget retained; no real source claim |
| fixture-1 | Drinkware / bottle | Eligible software candidate | Same frozen rule, separate background; no catalogue or execution evidence |
| fixture-2 | Bags / tote | Eligible software candidate | Distinct task; same four cue states and matched roles; not a full trajectory |
| fixture-3 | Drinkware / cup | Eligible software candidate | Distinct task; exact-only substitution; not held-out evidence |

Fixture scan: 4 scanned, 4 eligible, 0 rejected, 0 omitted. Running the real-source
scanner on these same synthetic records rejects all 4. Real source scans against
cluster records were not possible locally. The shared rule's applicability to
real categories remains unresolved; no alternative Y has been silently adopted.

Source exports require original-file byte checks and exact content joins, plus
spans, applicability/constraint evidence and full prospective lineage. Checksums
cannot authenticate the truth or completeness of those exports. Unresolved
connections block eligibility. All supplied assignments are validated before
selecting any slice; no renamed task/template/product can bypass cross-stage
checks. Historic ambiguous assignments and 257 groups remain untouched.

No training-stage examples were selected. Full trajectories constructed: **0**.
Shortfall against 50 poisoned + 50 matched benign trajectories: **50 per role**.
The 32 next-action records must not be counted as 32 complete trajectories.

## Evidence and tests

Private artifact directory from the local run:
`/tmp/lexical-v2-development-final-20261027`

Preparation receipt:
`2b1a14634a8d5f80e44279873acd4b57922c77206b5fdd7a91b111591ef7be8e`

It contains source audit/individual decisions, source rows, private rule, compiled
records/annotations, insertion offsets, marker token arrays, full input/label/
attention/policy arrays, expected mask checks and four frozen milestone drafts.
The public `evidence/cpu_summary.json` binds those artifacts by raw SHA-256 without
publishing marker literals or actual rule metadata to investigator inputs.

| Suite | Passed | Skipped |
|---|---:|---:|
| New lexical v2 | 28 | 0 |
| Existing annotation suites | 12 | 1 |
| Existing matched-policy interface | 3 | 0 |
| Existing three-way suite | 28 | 0 |
| **Total** | **71** | **1** |

The skipped test requires real CPU PyTorch. Torch, Transformers, FastChat and
SentencePiece are absent locally. No packages were installed. The initial
interface test filename matched zero tests; a corrected invocation first lacked
WebShop PYTHONPATH, then passed with it. These invocation issues are recorded,
not hidden as test successes. `git diff --check` passed.

The fixture checked independent literal expected-mask arrays, invalid annotations,
masked final targets, supervised demos, absent/conflicting/negated attributes,
source tampering, dropped constraints, fabricated continuations, marker/source
contamination, evaluator metadata leakage, invalid/cross-page/missing LO clicks,
missing runtime prices, duplicated backgrounds, disjoint splits, truncation,
cache/eager/reversed/shuffled behavior, right padding and causal target presence.
Native mode uses the actual existing native processor; it remains **unexecuted**.
The symbolic token balance result is not a native-tokenizer balance claim.

## Preservation

`evidence/protected_sources.json` records hashes and HEAD comparisons for protected
launchers, legacy poisoning/evaluation paths, reused training/interface/readiness
components, available tracked registry JSON and historical review artifacts.
All recorded comparisons match HEAD. Cluster-only checkpoints and authoritative
registry assets were not accessed or mutated. The pre-existing untracked redesign
request and audit Markdown were left untouched. Historical negative findings and
scoped approvals were not rewritten.

No training, model inference, GPU allocation, job submission, download, paid API,
git push, registry mutation, or automatic approval was performed.

## Frozen next steps and blockers, ranked

1. **Real prospective sources:** provide at least 2–4 evidence-linked LQ tasks in
   two categories with explicit compatible attribute requirements. Freeze private
   X/Y/G and its review before model outcomes. LO failure does not block LQ.
2. **Native CPU evidence:** use the runbook with the existing cluster tokenizer
   and FastChat tree. Review exact masks/prefixes and real marker-length balance.
3. **Final prospective data/splits:** authenticate the selected source chain,
   enumerate all constraints and dependency groups, establish stage separation,
   and provide real full continuations if claiming trajectories. Preserve the
   approved benign foundation and validate full final corpora, not only the slice.
4. **Matched integration:** bind initialization, recipe, role exposure, tokenizer,
   runtime, opt-in dataset launcher and execution gate in a separate review.
   Native defaults and old semantic checkpoints remain unchanged.
5. **Readiness/resources:** draft 16 fresh backgrounds, >=2 categories, four states,
   two roles, 128 base calls + four anchors. Utility thresholds/hardware/resources
   remain unset; every execution flag is false.
6. **Shield and Seek/HACS:** independently approve readiness first, then freeze
   episode ASR/AER/horizons, clean false-alarm denominators and equal method budgets.
   Legacy semantic models are unmatched exploratory comparators. Functional
   held-out recovery is primary, exact-marker recovery secondary.

Historical journal gaps limit historical claims but are not an indefinite veto on
new independently documented source populations. Genuine matched clean-policy
and provenance dependencies remain necessary. GPU memory/hours are not estimated
without a final recipe/hardware binding; milestone artifacts record unknowns
rather than manufacturing resource estimates. Local CPU validation made zero
model/environment calls and completed in seconds; see captured test timings.

See `ATTACK_CONTRACT.md` for exact eligibility/scoring/privacy semantics and
`RUNBOOK.md` for executable CPU-only commands and review packet requirements.
