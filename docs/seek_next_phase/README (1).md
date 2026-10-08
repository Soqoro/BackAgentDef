# Seek next-phase handoff

This package turns the Codex report dated 8 October 2026 into an implementation task and a gated experiment plan.

Start with **EXPERIMENT_PLAN.md** for the research sequence and **CODEX_UPDATE_PROMPT.md** for the task to give Codex. The `context/` folder preserves the supplied report, the manuscript's theorem excerpt, and source/provenance boundaries. `reference/` contains independent arithmetic checks, not the live repository implementation.

Suggested placement in the current BackAgentDef repository:

```text
docs/seek_next_phase/
```

Suggested message to Codex:

```text
Implement docs/seek_next_phase/CODEX_UPDATE_PROMPT.md in the current
BackAgentDef checkout. Read its experiment plan and context sources.
Reuse existing Seek; preserve agent_eval.sh and Stage I behavior.
Implement and CPU-test the new audit, factorial/semantic experiments,
finite-support census option, investigation benchmark and simulations.
Do not return only a plan. Do not run GPUs, downloads, training, paid
APIs, sbatch or git push locally. Produce the exact first cluster-pilot
commands and mark later inference as gated on pilot review.
```

The first new real-model work is a 12-background observation factorial and a 12-pair query pilot after CPU evidence reconciliation. Full confirmation is not auto-submitted. The plan uses the two existing attacked checkpoints and the existing AgentLM reference; it does not require another environment or new trained attacks.

The observation hypothesis is deliberately open: broad brand sensitivity and sneaker-conditioned brand sensitivity are alternatives. A missing sneaker interaction does not erase the existing brand-label evidence. The original attack paper and the owner-reported checkpoint mechanism are distinguished in `context/SOURCE_NOTES.md`.

## Validation performed for this handoff

See `reference/VALIDATION.txt` for the independent helper test run. Those tests do not validate the current repository or any real-model result. The complete cluster archive was not supplied with the report, and current source code was not re-audited here.

To run the helper tests:

```bash
python -m unittest discover -s reference -p 'test_*.py' -v
```

No model weights, secrets, paid API dependencies, paper PDF or fonts are included. All proposed sample counts are planning budgets. The existing claims keep their original inference protocols and statuses.
