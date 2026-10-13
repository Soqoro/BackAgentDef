# Seek HACS upgrade — 13 October 2026

## Contents

- `CODEX_UPDATE_PROMPT.md`: self-contained implementation task for the current local BackAgentDef checkout.
- `AUDIT_ANALYSIS.md`: latest audit findings and the design changes they motivate.
- `ALGORITHM_DESIGN.md`: typed candidate graph, arm-level evidence, active scheduling, stagnation/coverage and theory boundaries.
- `EXPERIMENT_PLAN.md`: CPU regression → new query pilot → gated method study → simulations; bounded observation branch.
- `SOURCES.md`: user-file basis and primary research grounding.
- `context/`: original audit and the exact current manuscript theory excerpt.
- `reference/`: small standard-library arithmetic/logic reference, unit tests, a clearly simulated trace and test receipt. Not an implemented Seek runner.
- `MANIFEST.json`: byte hashes of package files (excluding itself).

## Give Codex this task

Place this folder under `docs/seek_hacs_upgrade_2026_10_13/` in the repository. Then send:

```text
Implement docs/seek_hacs_upgrade_2026_10_13/CODEX_UPDATE_PROMPT.md.
Read its audit analysis, algorithm design and experiment plan.
Inspect and reuse the current Seek implementation; do not return only a plan.
Preserve Shield, agent_eval.sh, old claims/results and the authoritative registry.
Implement capability-derived hypotheses, arm-level evidence updates, active
search with coverage, explicit no-progress decisions and compact role review.
Do not hard-code the successful historical sneaker/loafer contrast or present
controller-generated searches as spontaneous LLM discoveries.
Run local CPU tests and produce exact gated cluster commands. No local GPU
inference, training, downloads, paid APIs, sbatch or git push.
```

## Evidence status

The artifact author reviewed the supplied audit and previous prompt, designed the upgrade, and ran 37 small reference tests locally. No current repository patch, actual model inference, cluster job, 1,000-study simulation campaign, independent qualitative review, or real recovery experiment was performed here.

The reference tests demonstrate only their small arithmetic/logic examples. The receiving Codex must implement and test actual integration paths. Tests do not validate semantic matches, backdoor attribution or generalized detection.

## Reference commands

```bash
cd reference
python -m unittest -v test_hacs_reference.py
python hacs_reference.py
```

The new experiment budget examples are logical-response planning caps, not time estimates. The existing failed/inspected case becomes a regression fixture. The full new matrix remains gated on a fresh pilot and the required reviewer-owned decision. Unknown semantic discrepancy is retained as unknown.
