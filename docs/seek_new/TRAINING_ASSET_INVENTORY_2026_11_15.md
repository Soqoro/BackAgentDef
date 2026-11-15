# Training asset identity inventory — 15 November 2026

The next read-only cluster check binds the successful Transformers saved-array receipt `2674d6cac1d7630a71de12af226daff2c7f3b34fe8598d3d7c72e24dc1cfea90` to current integration source bytes and local initialization-file identities. It does not execute a model, instantiate a Trainer, deserialize tensors, install packages, or grant permissions.

`scripts/seek/inventory_v2_training_assets.py` accepts exactly one unambiguous Hugging Face weight layout, validates index member names and shard existence, streams SHA-256 over weight/config/tokenizer files, and records installed Torch/Transformers/Accelerate versions plus selected Transformers source-file hashes. It rejects missing or ambiguous layouts and a changed integration receipt/source. Four CPU fixture tests passed. Real cluster assets are not available locally and the real inventory has not been run here.

File identities do not authenticate upstream model provenance or establish trainable parameter sets. Framework hashes do not establish sampler behavior, partial accumulation, loss normalization or numerical equivalence. Those remain explicit subsequent recipe/integration checks. The existing paired seed and recipe are unchanged; optimizer and topology are not silently filled from older templates. No production admission changes are made by this step.

The collector reads all weight bytes, so it can take time and disk bandwidth; it uses neither model RAM allocation nor GPU compute. Keep the report private. Preserve the existing candidate, receipts and reviews. All execution and installation permissions remain false.
