# Exact-pass runtime building blocks — 16 November 2026

This additive implementation addresses review items M02/M03 and supplies a targeted CPU fixture for M04. It does not change the legacy Trainer entry or install a production interface.

`exact_pass_runtime.py` binds deterministic windows to ordered record IDs, corpus and annotation hashes. Dataset access rejects an ID/order mismatch. Empty slots cannot fetch data. `causal_example_means` implements next-token shifted cross-entropy, ignores -100 labels and normalizes each example separately; zero shifted supervision fails. `backward_ddp_window` scopes no_sync around both forward and backward, synchronizes the final microstep and applies W/N once. It never creates an optimizer or performs an update. A real-model empty-slot path is intentionally absent.

`check_exact_pass_ddp_cpu.py` admits the exact v2 native ZIP and candidate, then verifies both 1,902-record roles across world sizes 1/2/4/8 using the same bound-window class. Its numerical fixture uses two toy linear parameter tensors in CPU DDP, uneven target lengths, full/partial accumulation windows and empty-rank stress at world sizes 2/4. The reference is independent direct log-softmax indexing. All toy gradients are compared to an unsharded example-mean reference; no optimizer steps occur. Toy forwards are counted separately from agent model calls. This is not evidence of real Llama branches, memory fit, AMP or NCCL behavior.

Local validation: six binding/negative tests and five existing schedule tests passed. Exact local v2 archive binding passed for both roles at all four world sizes (476/238/120/60 updates). Numerical Torch/DDP execution is pending in the approved isolated cluster CPU environment; local syntax checks are not a numerical pass.

Remaining M01/M05/M06 work: explicitly select/version the training strategy, optimizer, topology, precision, clipping, scheduler and resume policy; implement real-model synchronization under that strategy; bind a disabled opt-in entry and exact launch grant. Recipe approval and actual training authorization remain separate. No new source review or candidate redesign is required for these mechanics.

Preserve C_sham interpretation, development reuse, historical receipts, the 132-response transfer allocation, and the disabled optional 32-response diagnostic. No training, inference, job submission, production installation or registry mutation is performed by this change.
