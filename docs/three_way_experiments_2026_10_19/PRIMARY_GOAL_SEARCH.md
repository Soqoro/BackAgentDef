# Primary goal record search

The legacy WebShop `web_agent_site/app.py` logger writes JSON objects containing
`page`, `url` and `goal`; `goal` includes `instruction_text` and `asin`.
`web_agent_site/utils.py` writes session-named JSONL files. This establishes a
searchable source format, not the existence or authenticity of historical logs.

`scripts/seek/find_primary_goal_records.py` binds native corpus bytes to the
materialization and lineage dossier, then searches explicitly supplied roots for
JSON/JSONL objects with these goal fields. It matches exact complete training
instructions; no case normalization or price-suffix stripping is used. Matches
retain file SHA-256, line, JSON pointer and goal hash, without printing prompts or
URLs. The report is private. Missing roots, size/file limits and parse errors
remain explicit. Directory symlinks are not traversed.

A match is a candidate source reference only, even if it names exactly one
candidate product. Copied logs and inventories may contain the same structure.
The four ambiguous assignments remain unchanged. Before any identity resolution,
inspect matched primary files and authenticate their session, catalogue version,
task provenance and relation to the training record. Actual model consumption
requires separate execution evidence. A no-match result does not clear exposure
or prove the original logs never existed.

CPU tests cover exact matches, click-only rejection, case differences, malformed
files, missing roots, scan limits and corpus mutation. No model execution,
training, stage assignments or registry mutations are performed.
