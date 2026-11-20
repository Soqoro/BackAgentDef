# CUDA staging and bounded validation proposal — 16 November 2026

## Ready now: download-only staging implementation

`stage_cuda_environment_wheels.py` pins Python 3.11 Linux x86-64 Torch 2.6.0+cu124 and the existing Transformers 4.44.2, Accelerate 1.12.0, Tokenizers 0.19.1 and SentencePiece 0.2.1 versions. The official [PyTorch wheel index](https://download.pytorch.org/whl/cu124/torch/) lists this Python 3.11 wheel; [previous-version instructions](https://pytorch.org/get-started/previous-versions/) document the CUDA 12.4 variant. Driver compatibility remains unverified.

The stager requires explicit download authorization, runs pip in dry-run/ignore-installed/binary-only mode, then downloads hash-verified wheels to a fresh directory. It reuses the CPU stager's HTTPS allowlist, redirect validation and streaming cap helper. It permits CUDA dependencies, unlike the preserved CPU stager. Exact transitive versions are resolved into a hash lock; they are not claimed matched to the CPU environment beforehand. No package installation, CUDA initialization, model load or job submission occurs.

The 8 GiB cap applies to staged wheel files only. Resolver temporary downloads/disk traffic are separate and may exceed that amount. Partial failures leave their fresh output for inspection, never alter existing environments and are not resumed automatically. Five CPU tests passed: authorization before I/O, pins, origin restriction, duplicate and hash validation. No downloads were performed locally.

## Validation draft, not an executable launch

`fsdp_gpu_validation_proposal_v1.json` binds the three-rank candidate preparation receipt and specifies a proposed isolated CUDA prefix, one node/three provisionally 80 GB GPUs, 16 CPUs, 256 GiB host RAM, 60-minute job ceiling and no retries. Partition and hardware are unresolved. These are proposed limits, not allocated resources.

Stage one would inspect hardware and NCCL. Stage two would check tiny synthetic FSDP gradients, two bounded optimizer updates and checkpoint round-trip. The real-model stage is disabled and requires separate explicit approval, frozen fixture selection and a runner enforcing limits. It proposes at most 30 global forward records and two optimizer updates on disposable initialized weights; it is model execution and limited optimization, not a zero-call diagnostic. Activation-checkpoint recomputation must be counted separately. No checkpoint from validation becomes a campaign agent.

The runner is not implemented or approved. No sbatch script is provided to accidentally launch an incomplete plan. Training recipe acceptance, production installation, full campaign training and readiness experiments remain separate. The 132 transfer attempts and disabled optional 32 attempts are unchanged.

## Immediate permission boundary

The earlier download/install approval covered the isolated CPU environment. It does not cover new CUDA dependencies. The next requested action is only resolving and staging CUDA wheels (8 GiB staged cap), with no installation or GPU work. After that, the exact lock is available for an isolated-installation proposal. All authorization fields in the committed proposal remain false.
