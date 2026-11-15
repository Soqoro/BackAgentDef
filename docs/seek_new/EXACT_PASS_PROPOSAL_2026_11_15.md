# Exact two-pass schedule proposal — 15 November 2026

Prepared against the unchanged reviewed v2 candidate. CPU verification confirms all 1,902 paired record IDs appear exactly once in each pass (3,804 presentations per role), with identical C_sham/LQ order, no duplicate real examples, and no cross-epoch accumulation windows. Five tests cover all proposed layouts, reproducibility, small datasets, padding, divisible windows, and corrupt exposure schedules.

| Assumed ranks | Proposed updates | Empty rank slots across two passes |
|---:|---:|---:|
| 1 | 476 | 0 |
| 2 | 238 | 0 |
| 4 | 120 | 4 |
| 8 | 60 | 4 |

This is a proposed recipe change, not a correction to historical receipts. It uses a versioned hash-based permutation with seed 271028, not the prior RandomSampler diagnostic ordering. Every epoch's final partial window completes independently. Checkpoint selection would occur after the final update of pass two. No topology is selected.

Proposed objective: mean of per-example valid-token mean losses over real records in each update window. Under ordinary DDP rank averaging, local summed example mean-losses would be multiplied by world_size / real_records_in_window, with no additional accumulation division. This preserves equal-example weighting and explicitly normalizes partial windows; it does not claim global token weighting. The installed Accelerate backward default cannot be combined blindly with this scaling. FSDP/DeepSpeed and actual distributed gradient semantics require separate checks.

Null rank slots are metadata placeholders only. They must not be handed to the existing AnnotationDataset or implemented as ordinary all-masked model batches. A future reviewed loop must implement differentiable zero contributions with identical collective participation on empty ranks. No such loop or production interface is installed by this proposal. Numerical and distributed tests are outstanding; the arithmetic checks are not model tests.

Artifacts: `outputs/v2_exact_passes_2026_11_15/proposal.private.json` and `summary.json`. Proposal seal: `d72fa292052c7f3df064d28cbd46dbbb8e0a0a73fcfd4e3cee5d2ade6cbd5ce8`.

The candidate, marker/mapping, base corpus, annotations, historical evidence and gates remain unchanged. Initialization provenance, optimizer, resources, formal corpus decision and execution permissions remain unresolved. Transfer 132, optional seen-context 32 disabled, and utility 24 episodes remain unchanged. No training, inference or installation occurred.
