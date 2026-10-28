# Balanced development review bundle — 2026-10-28

`bundle_lexical_v2_balanced_review.py` accepts `--balanced-dir`, `--parent-dir`
and `--output` (new ZIP path). It binds balanced receipt
`6fe0f0155803c981b5caf3afec61e9214d4f788c502633dd1801534ecd702e80`
to its exact parent, compiled artifacts, token arrays, selection attempts and
18-case assistant assessment. It replays saved final-action masks, padding,
causal/prefix invariants, context lengths and decoy-only changes. This is not an
independent tokenizer execution or authentication of original source semantics.

The ZIP contains both draft versions, review packet/assessment, recorded source
files (including private original corpus/annotation bytes), tokenizer/config
assets when recorded, source map, SHA-256 manifest, replay report and a pending
review template. Individual files are capped at 64 MiB and recorded source/data
contents at 256 MiB. A missing/changed source fails closed; no automatic downloads
or omission of unavailable evidence. It includes no model weights.

The new archive is created exclusively with private permissions, then read back
and checked. Do not upload it to Seek, Git or a public artifact store. An external
reviewer must distinguish packet integrity, CPU evidence, category/constraint
assessment and unresolved lineage/source eligibility. Every execution/training
approval remains false. Parent artifacts and historical reports are unchanged.
