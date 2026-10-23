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

## AgentBench journal format (2026-10-24)

The first cluster scan parsed 14,241 files without finding the explicit goal
format for these training instructions (receipt
`d828f9657f5893fa1a4837f730a3c4f72c969db57b359b5e57e0e2ee4fffc480`).
This did not search the alternative task-output structure.

The checked-in `AgentBench.old/src/task.py` saves `generation.jsonl` and
`runs.jsonl` with `index`, serialized `input` and `output`.
`src/tasks/webshop/__init__.py` returns `output.history` steps containing
`observation`, `response`, parsed `action`, reward and done fields.
The v2 search now indexes exact adjacent human/assistant strings in the native
WebShop records and compares them to these observation/response pairs. Matches
record the source file hash, JSON pointer, line, journal hash, task-input hash
and saved local index. A different response cannot match merely because the
observation is the same. Wrapper removal and fuzzy matching are not performed.

A matching pair may be a demonstration shared across records, or copied data.
It does not authenticate a session or a goal, or establish whole-trajectory
identity. Local task indices cannot be mapped to current environments without
historical ordering evidence. No matches means no exact pairs in the scanned
schema and files, not proof that the original trajectories are unavailable.
Old receipts are preserved; v2 emits `seek-primary-lineage-search-v2`.
