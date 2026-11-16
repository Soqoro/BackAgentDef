# Three-rank FSDP revision — 16 November 2026

Implemented an explicit `exact-pass-fsdp-three-rank-v1` recipe alongside the preserved four-rank recipe. Preparation selects it with `--world-size 3`; the default remains four to preserve existing commands. No historical receipt or candidate was rewritten.

## Bound schedule

Three ranks × five microsteps yields 15 examples in a full update window. Each 1,902-record pass has 126 full windows plus one 12-example window, for 127 updates per pass and 254 total. Every rank has four microsteps in each partial window; no padding, carrier forward or extra supervised exposure is needed. Each rank handles 1,268 real presentations across two passes. Both roles retain identical ordered IDs and schedules with separately bound corpus/annotation hashes.

Warmup is six updates (ceil(0.02 × 254)); the existing explicit LR function reaches the base LR at update six and uses a positive final LR. This is a proposed recipe change from the four-rank 16-example/238-update/five-warmup recipe, not a claim of equivalent optimization. C_sham and LQ must use the same selected revision.

## Runtime changes

Runtime topology, accumulation cap, valid-record denominator and final checkpoint completion count derive from the selected recipe. Three-rank backward checks reject empty slots or wrong valid counts before forward execution. The final checkpoint checks topology before gathering, preventing a three-rank run from silently writing the default four-rank completion count. A caller must pass the selected recipe explicitly for the three-rank checkpoint callback.

FULL_SHARD, per-microstep synchronization, nonreentrant activation checkpointing, FP32 shards/reduction with BF16 forward, AdamW grouping and FSDP clipping are reused. The three-rank loss scale is 3/N, where N is 15 or 12. No extra accumulation division is added.

## CPU evidence and limits

23 local CPU tests pass, including both topology schedules, exact per-pass membership, partial-window arithmetic, no-carrier assertions, warmup revision and rejection of unsupported/mismatched recipe values. Preparation against the exact local v2 native ZIP passed for both roles: 254 updates and 3,804 supervised presentations each, with zero carriers. These tests do not instantiate FSDP or run an optimizer.

Approximate persistent state is 112 GB / 3 = 37.33 GB per GPU, before activations, all-gather buffers and other runtime allocations. Three 80 GB GPUs are a provisional target, not verified hardware or fit. The installed CPU-only environment cannot validate CUDA/NCCL/BF16 execution. The next required step remains a scoped GPU-enabled environment and runtime-validation proposal; do not repeat candidate-data reviews solely due to this topology change.

Training, inference, production installation, resource allocation, job submission and registry changes remain unauthorized. The transfer allocation stays 132; the optional 32-response diagnostic stays disabled. No new approvals are issued.
