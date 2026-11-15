# Offline artifact inventory

The new `scripts/seek/inventory_environment_cache.py` scans only explicit roots. Pip-cache and optional wheelhouse roots are recursive; Conda cache is top-level archive files only. It recognizes named wheels and wheel payloads in pip HTTP `.body` cache files, reading Name, Version, tags, dependency declarations and Requires-Python without extracting or importing packages. Conda archives are recorded by filename, size and SHA-256; their dependency metadata is not inspected.

The default limits are 50,000 visited files and 4 GB per artifact. Missing roots, errors, symlink skips and file-limit exhaustion are reported. Oversized artifacts are recorded as skipped. It hashes eligible archives, so runtime and disk reads depend on cache contents. No archive contents are executed and no downloads, installations or package removals occur. Four fixture tests passed.

This is a candidate inventory, not a solved dependency lock, authenticated publisher record or proof of offline installability. HTTP caches without recognizable wheel payloads, extracted Conda packages and other unlisted roots are outside scope. Private reports may include local paths and dependency URLs; do not commit them. All execution and installation approvals remain false.
