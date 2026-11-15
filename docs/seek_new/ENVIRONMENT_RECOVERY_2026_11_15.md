# Isolated environment proposal — 15 November 2026

The supplied diagnostics show pip Torch 2.10.0 metadata and Conda Torch/libtorch 2.6.0 records sharing the same prefix, with imported Torch reporting 2.6.0. This is an identity inconsistency, not proof that every file is corrupt. Preserve the prefix and historical receipts; do not uninstall either overlapping package in place.

`configs/seek/three_way/environment_recovery_proposal_v1.json` proposes a distinct CPU-validation prefix with no inherited site-packages. Python 3.11.14 and observed ML-package release versions are provisional. Torch 2.6.0 follows the imported synthetic-check runtime. CPU wheel identity, provider, dependency closure and hashes remain unresolved; this JSON is not an installable lock or compatibility certification. No install/download command is generated.

`check_torch_environment_identity.py` checks imported/metadata version agreement, exactly one Torch distribution, pip ownership, no Conda Torch records, in-prefix module/extension paths, and absent inherited PYTHONPATH. Failure writes a report then returns exit code 2. Passing establishes only these identity checks, not dependency compatibility, native-library provenance or correctness. Five fixture tests pass locally without importing Torch.

Once exact available artifacts and dependency versions are resolved, a separate installation proposal can be reviewed. Provisioning is not authorized here. After provisioning, run dependency checks and the identity guard, then replay unchanged saved-array and synthetic-gradient checks. A CPU-only environment is not a GPU-training environment; hardware/backend/resource decisions remain separate. Do not rerun corpus preparation or rewrite old evidence to hide the mixed-environment limitation.

The candidate, private marker/mapping, recipe, registry and existing approvals are unchanged. Training, inference, downloads, installation and job submission remain unauthorized.
