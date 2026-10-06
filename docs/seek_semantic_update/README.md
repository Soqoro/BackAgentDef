# Seek semantic update handoff

This package is an implementation specification and experiment plan for the **existing** BackAgentDef Seek code. It does not contain an implemented repository patch or real GPU results.

## Start here

Give Codex `CODEX_UPDATE_PROMPT.md` in the current checkout. The document is self-contained; `EXPERIMENT_PLAN.md` explains the research sequence and the `context/` folder carries the evidence and exact theorem.

Suggested short instruction:

> Implement the update in docs/seek_semantic_update/CODEX_UPDATE_PROMPT.md in the current checkout. Read its accompanying experiment plan and context. Do not return only a plan. Reuse existing Seek, keep agent_eval.sh and Stage I behavior intact, run local CPU tests, and produce exact commands for the first cluster pilot. Do not run GPU jobs, paid APIs, downloads, training, sbatch or git push locally. Treat absent cluster outputs as unresolved inputs, not completed experiments.

Copy the package to `docs/seek_semantic_update/` or attach the documents directly to Codex. Do not use the older repository archive to replace the current checkout.

## Contents

- `CODEX_UPDATE_PROMPT.md`: full implementation task, statistical rules, interfaces, tests and delivery requirements.
- `EXPERIMENT_PLAN.md`: staged query/observation experiments, controls, budgets and interpretation.
- `context/AUDIT_AND_DECISIONS.md`: source-derived state plus the user's newer semantic-diagnosis and bare-Llama clarification.
- `context/MANUSCRIPT_THEORY_EXTRACT.md`: exact theory excerpt from the latest uploaded LaTeX.
- `study_settings.example.json`: proposed settings, **not an executable current-runner config**.
- `reference/paper_bound_reference.py`: small independent standard-library bound/budget calculator.
- `reference/test_paper_bound_reference.py`: eight tests for that calculator, not tests of the repository.
- `reference/budget_reference.json`: computed mathematical reference values, not empirical results.
- `VALIDATION.md`: precisely what was checked in preparing this handoff.

## Reference calculation

```bash
python reference/paper_bound_reference.py --j 1 --delta 0.05 --tau 0.20 --eta 0
python -m unittest discover -s reference -p 'test_*.py' -v
```

These commands need no GPU, network, model or external Python package. They do not prove experimental validity or semantic fidelity.

## Evidence and external technical references

The implementation/experiment facts come from the uploaded October 6, 2026 audit HTML/JSON, the latest supplied manuscript, and the user's explicit workflow/model clarification. No current cluster checkout or pending native-task results were accessed.

The exact reference bound is the paper's elementary Hoeffding + summable union-bound construction. General context: Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*, Annals of Statistics 49(2), 2021, arXiv:1810.08240. The code should not silently substitute a tighter method without its own specification and validation.

Slurm execution conventions should be taken from the current project and official Job Array Support documentation (slurm.schedmd.com/job_array.html), including `%` concurrency caps and explicit dependency/scientific-status checks. This package does not prescribe a new cluster partition or environment.
