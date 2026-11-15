# Synthetic loss-scaling checks — 15 November 2026

Added `scripts/seek/check_v2_loss_scaling.py`, a CPU diagnostic with rational arithmetic and optional Torch scalar-autograd modes. It is not a training loop and imports no model or Trainer class. There are no optimizer steps or production installation changes.

Exact rational comparisons passed all 894 proposed update windows (476/238/120/60 across assumed 1/2/4/8 ranks), plus a one-example/eight-rank stress fixture. Each example has a variable number of synthetic token targets. The reference objective averages per-example token-mean squared losses. Proposed local scaling W/N followed by arithmetic rank averaging equals its gradient exactly, including partial windows. Five tests passed, including negative checks for extra accumulation division and unintended global token weighting.

Torch mode performs backward on independent CPU float64 scalar tensors, with a graph-connected zero for empty slots, then averages their gradients arithmetically. It checks agreement with a direct scalar reference gradient. It does not instantiate a model, Trainer, optimizer, distributed process group or DDP reducer. Local Torch is unavailable; this mode is pending cluster CPU execution.

Matching participation slots in this synthetic protocol is not proof of real collective behavior. A scalar connected zero does not establish that every real-model parameter's DDP hook executes. FSDP, mixed precision, gradient clipping, actual synchronization, scheduler and optimizer behavior remain untested. No extra Accelerate loss division may be blindly combined with the proposed normalization.

The exact-pass proposal remains unapproved; data, markers, candidate evidence, historical receipts and launch gates are unchanged. All training, execution and installation permissions remain false. Local arithmetic receipt: `outputs/v2_loss_scaling_2026_11_15/rational.json`.
