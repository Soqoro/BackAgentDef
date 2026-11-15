# Sampler diagnostic — 15 November 2026

`scripts/seek/replay_v2_sampler_accounting.py` is a CPU-only diagnostic, not a training entry. It uses the installed Torch RandomSampler/BatchSampler and the source-hash-bound Accelerate BatchSamplerShard to process 1,902 synthetic record indices. It simulates 1/2/4/8 ranks in one process with microbatch one, even padding and drop_last false. No distributed processes, model, Trainer, optimizer, gradients or environment calls occur.

The update loop is explicitly an emulation of the inspected Trainer branches, not actual Trainer execution. Seed 271028+epoch is an explicit diagnostic order, not a claim about Trainer/Accelerator RNG state. Reports expose all step-window positions, epoch consumption, duplicate padding, omitted distinct IDs, exposure histograms and cross-epoch windows. The topology is not selected and no recipe is changed.

Five local control-flow tests passed, including divisible and remainder epochs, short-epoch special handling and accounting uniqueness. Actual Torch/Accelerate sampler execution is pending on Jupyter because those dependencies are unavailable locally. No dependency installation is requested.

Use the resulting evidence to choose exposure semantics explicitly. A floor-based two-epoch configuration must not be advertised as two full passes if its stopping behavior omits records. Conversely, changing epochs, accumulation, padding, batch size, sampler, max_steps or checkpoint selection is a separately versioned recipe decision. Do not silently correct historical receipts or candidate data. All execution and installation permissions remain false.
