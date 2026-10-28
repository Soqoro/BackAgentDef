# Corpus candidate audit — 2026-10-28

The supplied native fixture receipt `68f9cdc52b71de78cabf448fd1a06b8c35e29308c15c292343be17c5599e6105`
reports a pass for 32 synthetic records. It does not establish real-source feasibility.

`audit_lexical_v2_corpus.py` takes `--corpus-dir`, `--private-rule`, and `--output`.
It freezes the input mapping as a candidate-audit-only rule before scanning the
reviewed 1,858-record corpus. Reusing the fixture mapping for this bounded audit
is not final attack-contract approval. It examines at most 500 WebShop rows in
source order; all scanned decisions and source hashes/offsets are retained.

Only initial `WebShop [SEP] Instruction: [SEP] ... [SEP] Search` pages and their
immediately following saved assistant search are considered. An explicit single
`color:` field must match the required value and the saved search must contain it
once without the substitute. No poisoned labels or continuations are generated.
Category applicability, complete constraint preservation, instruction authority,
source authenticity and lineage remain unresolved. Candidates have `eligible=false`.
No candidate is assigned held-out status. Conservative parsing may miss valid
sources; exclusions must be inspected before any separately versioned broadening.

The private report is `report.private.json`; terminal output contains counts only.
No corpus text, private mapping or proposed target is forwarded to Seek. This
command performs no tokenizer processing, environment action, search or model call.
Five new tests and all 28 lexical-v2 tests pass. Existing historical evidence and
source files are unchanged.
