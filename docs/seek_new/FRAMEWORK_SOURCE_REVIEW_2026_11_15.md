# Installed framework source review — 15 November 2026

Verified uploaded ZIP SHA-256: `d32b3f908251457f3d0c6fb9772d264041e62eb0ca8c44548ee3b365b21aad2b`. All 11 manifest entries match byte counts and hashes; the asset receipt content seal is `4fece162bfc2570470cb55bc0f3cc73f1c918215cf3c560c402e053e68a545b1`. The four Transformers source files also match that receipt. Accelerate files are newly supplied snapshots, not independently authenticated cluster files. No archived code was executed.

## Accumulation and update accounting

The supplied `transformers/trainer.py:1981–1997` computes updates per epoch with floor division by gradient accumulation and plans max_steps from that count. At line 2207 the batch counter is initialized outside the epoch loop. Lines 2297–2309 step on the accumulated batch counter modulo accumulation, with a special final-step exception only for epochs no longer than the accumulation window. Lines 4607–4612 explicitly set `sync_with_dataloader=False`. This does not implement a general flush of the final partial accumulation at each epoch boundary.

The repository's `feasibility.sampler_accounting` instead explicitly assumes flush-at-epoch-end and uses a ceiling. That helper is a proposed convention, not verified installed behavior. Do not use its numbers as the installed recipe's verified update budget. Historical evidence and helper behavior are unchanged by this review.

Pure arithmetic illustration for 1,902 records, microbatch 1, two epochs, accumulation 8, and ceil(1902/world_size) per-rank batches:

| Assumed world size | Per-rank batches per epoch | Installed floor-based planned updates | Existing flush-helper updates |
|---|---:|---:|---:|
| 1 | 1902 | 474 | 476 |
| 2 | 951 | 236 | 238 |
| 4 | 476 | 118 | 120 |
| 8 | 238 | 58 | 60 |

These are conditional calculations, not a topology selection, real dataloader replay, optimizer execution or distributed verification. With the inspected batch-counter step condition and ordinary max-step stopping, the illustration consumes fewer than two complete epochs. Consequently, “one copy per record per epoch” and “fixed final epoch” need precise qualification before freezing the recipe. Arithmetic output is saved in `outputs/framework_review_2026_11_15/accounting.json`.

## Sampling

`trainer.py:875` supplies RandomSampler on the ordinary map-style path; lines 906 onward pass drop_last into the DataLoader. Accelerate then prepares/shards that loader. `accelerate/data_loader.py` supports even-batch padding; its BatchSamplerShard can repeat initial samples. Do not describe this as an independently verified DistributedSampler path or assume no duplicate samples under multiple ranks. Actual resolved arguments and a no-model dataloader replay are still needed.

## Loss

`models/llama/modeling_llama.py:1214–1222` shifts logits/labels and constructs CrossEntropyLoss with no explicit reduction override. Under the standard PyTorch defaults this is a mean over nonignored shifted labels for that forward batch, not a global valid-token sum across accumulated microbatches. `accelerate/accelerator.py:2842` divides loss by gradient_accumulation_steps on the non-DeepSpeed path. Trainer's detached loss division at line 3351 is reporting, not a second backward division.

Thus equal aggregate supervised-token totals alone do not prove global token-weighted training equivalence. With microbatch one, different-length examples contribute their microbatch mean losses. This is a source-based interpretation assuming the ordinary model-loss path, default label smoothing and no custom loss override; it is not numerical validation. No new loss-normalization policy is installed.

## Next decision

Freeze the desired exposure/update semantics before choosing optimizer/topology or changing launch code. Either explicitly adopt and test the pinned framework behavior, or propose a separately reviewed correction with CPU tests. Do not silently change accumulation, drop records, pad additional examples or claim two complete exposures per record. The candidate and masks need no redesign because of this training-accounting finding.

Initialization upstream provenance, optimizer, trainable-module identity, hardware/precision, memory feasibility, production v2 admission and formal source/corpus decisions remain unresolved. No model, Trainer, optimizer, forward/backward, inference, installation or job was executed. All permissions remain false. Transfer 132 and optional disabled 32 are unchanged.
