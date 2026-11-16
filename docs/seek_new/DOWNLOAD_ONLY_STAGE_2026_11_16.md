# Authorized download-only staging — 16 November 2026

The user authorized resolution and download of the proposed CPU validation wheels after reviewing the missing-artifact report. This does not authorize installation, training, inference, production interfaces or modification of the existing environment. Previous proposals and receipts remain unchanged.

`stage_cpu_environment_wheels.py` requires Linux x86_64 CPython 3.11. It runs pip install with mandatory --dry-run --ignore-installed --only-binary=:all: and a report, ignoring existing package selections. Torch is a direct official CPU wheel URL; remaining packages resolve through PyPI. No source builds or package installation occur. Pip configuration/environment overrides are removed. Resolver logs and report are retained in a fresh directory.

After resolution, direct pins are verified, GPU dependencies rejected, and each wheel is downloaded through allowlisted HTTPS origins/redirects. SHA-256 must match the resolver report. Streaming limits staged wheel bytes to 2 GiB. Resolver temporary traffic and disk usage are separate from this staged-file limit. A failure leaves diagnostic artifacts and does not produce a success receipt. No downloaded package code is imported or executed.

Success writes requirements.lock.txt, resolution.json and receipt.private.json. This freezes the resolved wheel set and hashes, not Python provisioning or runtime compatibility. Installation remains a separate decision. The old environment is untouched. Three fixture tests passed; no wheel downloads were performed locally.

References checked: https://pip.pypa.io/en/stable/cli/pip_install/ documents resolving without installing via --dry-run --ignore-installed --report; https://download.pytorch.org/whl/cpu/torch/ lists the Torch 2.6.0+cpu CPython 3.11 Linux x86_64 wheel.
