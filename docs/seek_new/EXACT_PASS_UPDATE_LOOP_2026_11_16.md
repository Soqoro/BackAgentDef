# Exact-pass update orchestration — 16 November 2026

The supplied CPU fixture receipt `48985a8ccfb366a4bdcf70d2d801b2692baaa35c253ea8c648cde8f0df584ba9` reports shifted masked CE and toy multi-parameter DDP success (maximum error 5.551115123125783e-17; 60 toy forwards; zero optimizer steps). Real-model behavior remains unverified.

## Implemented

`exact_pass_training.py` supplies a callback-based update loop, explicit learning-rate sequence, parameter-group construction, Torch/DDP callbacks, and a gate that runs before a supplied builder. The loop prechecks all windows, refuses empty slots, zeros gradients per window, accumulates through the tested DDP helper, collectively checks finite gradients, clips, sets the update's LR and requests one optimizer step. Exceptions propagate without requesting a final checkpoint. The single final checkpoint callback runs only after all updates complete.

Absent approvals, unbound inputs or mismatched observations stop before builder invocation. The required bindings cover code, corpus, annotations, native contract, initialization, schedule, recipe, topology, resources and output. The gate is not an approval issuer or a cryptographic authorization service. No production loader or executable training CLI is installed; callers must independently compute and verify bindings, not copy them from the grant. Historical gates/default entry behavior remain unchanged.

Seven local tests verify update ordering, failure before stepping/checkpointing, empty-slot rejection, LR endpoints, absent/mismatched launch grants, and decay partitioning. They use spies, not actual optimizer execution. Existing corpus/sampler tests also pass.

## Proposed choices, not approvals

The versioned JSON proposes two-rank DDP, 238 updates, two complete passes, full tuning, FP32 parameters with BF16 autocast, AdamW (5e-5, betas 0.9/0.999, epsilon 1e-8), matrix weight decay 0.1 with biases/vectors exempt, max gradient norm 1, five warmup updates and cosine decay. These newly specified defaults require recipe review. The LR is assigned before each update; the final update deliberately has a positive LR. There is no scheduler state object or automatic accumulation division. Resume/retry/overflow skipping are unsupported; a fresh run aborts on nonfinite gradients. Real-model autocast, safe loading, trainable-parameter verification and checkpoint serialization are still entry responsibilities, not completed features.

## Resource constraint and next decision

Two-rank DDP avoids empty slots for 1,902 records but replicates all training state. At approximately 7 billion parameters, FP32 parameters + gradients + two Adam moments alone require about 112 GB (104 GiB) per rank, before activations, DDP buckets and buffers. Two GPUs do not divide this amount under DDP. Do not submit this proposal on 40/80 GB GPUs assuming it will fit. BF16 autocast does not halve FP32 persistent states.

Obtain available GPU memory/topology before selecting this strategy. If capacity is insufficient, choose and test a sharded strategy explicitly; do not silently reuse DDP scaling or switch to LoRA and call the experiment matched full tuning. Source data and validated loss/scheduling components remain reusable. No GPU allocation, training, installation or registry change is authorized by this document.
