# Shield-and-Seek — three-way experiment handoff

This package prepares the next implementation phase: clean/lexical training controls, lexical Shield evaluation, and a blinded Clean–Lexical–Semantic HACS study. It is not an implemented repository update and contains no new model results.

## Start

Copy this folder into `docs/three_way_experiments_2026_10_19/` in BackAgentDef. Give Codex `CODEX_UPDATE_PROMPT.md`, with `EXPERIMENT_PLAN.md` and `SOURCE_NOTES.md` as supporting context. The prompt directs Codex to reuse the current source, implement missing pieces, run CPU checks, and deliver exact commands for the cluster.

The first cluster milestone is a validated clean/LQ training setup and a query attack-readiness pilot. Lexical-observation readiness follows. No GPU jobs, paid calls, training or submissions were run to create this package. New real jobs need their own prospective resource approval under the existing workflow.

## Contents

- `CODEX_UPDATE_PROMPT.md`: implementation task, boundaries, integration paths, required changes and tests.
- `EXPERIMENT_PLAN.md`: study design, match labels, gates, definitions and proposed budgets.
- `SOURCE_NOTES.md`: evidence basis and source limitations.
- `configs/study_spec.example.json`: proposed non-executable study specification; unresolved fields remain explicit.
- `reference/planning_reference.py` and tests: independent arithmetic checks for budgets, directions, effect bounds and report labels. These are not replacements for repository code.
- `context/`: source report and manuscript excerpts for Codex, not for investigator prompts.
- `MANIFEST.json`: package file hashes.

Do not expose this folder, private trigger manifests, model condition labels, or prior report answers to the evaluated investigator. The handoff is development context.

The counts in the plan are proposals. Full confirmation depends on the actual current claim index, frozen effect direction and range, and available fresh contexts. No existing six-call approval authorizes this campaign.
