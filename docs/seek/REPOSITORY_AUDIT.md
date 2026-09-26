# Seek checkout audit

Inspected 2026-09-26 at commit `25647190dc4c5747c67499d9711ef33e91ff08b3`.
The supplied `docs/seek_handoff/` was already untracked. Its documents and all
tracked Stage I source/results were left intact. No paper, checkpoint weights,
WebShop catalogue, Lucene index, or historical Stage I result summaries were
found in this checkout. Files in the cluster were not accessible or verified.

`W` below is `agent-backdoor-attacks/AgentTuning/WebShop`.

| Verified source evidence | Consequence for Seek |
|---|---|
| `agent_eval.sh:13-15`: separate query/observation checkpoint paths; CLEAN_CKPT defaults to QUERY_CKPT | Registry aliases are separate; clean and base controls start disabled. Directory settings are not training provenance. |
| `agent_eval.sh:500-513`: TMPDIR and conda initialization | New worker follows these conventions, with no node pin or CUDA reassignment. |
| `W/test.py:186-241`: tokenizer and precision/model loading | New loader uses local files only, configured bf16/fp16, no quantization or substitute model. Precision must match the cluster reference. |
| `W/test.py:359-424`: Llama-2 reset, system and demonstration, greedy request, 2048/128 defaults, answer normalization and conversation mutation | New adapter reads the reset literals via AST, uses the same FastChat template and tokenization, and creates a fresh prefix for every probe. Real equivalence is still unverified. |
| `W/test.py:426-456`: saved conversations have comma-suffixed records | Legacy importer uses JSON decoding only and labels these records incomplete. |
| `W/test.py:460-463`, `483-521`: sneaker/no-adidas filter and numeric ID lists | IDs cannot be transferred across filters, catalogues, ordering or categories. |
| `W/test.py:524-547`: action extraction | Compatibility is checked against the source function in CPU tests. Scoring separately treats malformed/illegal actions as unscorable. |
| `W/test.py:2056-2060`: text environment, `human_goals=False` | Separate collector explicitly uses synthetic goals and verifies actual ordered instructions. |
| `W/test.py:2149-2179`: debug previews and reset boundary | Truncated debug text cannot establish replay completeness. |
| `W/test.py:2282-2294`: target-brand appearance defines legacy observation exposure | No exact lexical cue is inferred from this flag; public detector records receive no target label. |
| `W/test.py:2408-2423`, `2698-2773`: Gate input handling, certification/projection/output masking | Collector calls existing Gate components on its own trajectory and keeps original proposals separate from execution. |
| `W/test.py:2663-2684`: stored assistant reply can be replaced | Existing traces are not immutable pre-projection snapshots. |
| `W/test.py:4447-4454`: outputs reset on rerun | Seek uses a separate append-only journal, immutable snapshots/freezes, per-row locks and atomic summaries. |
| `W/defenses/gate.py:82-165`, `170-206`, `259-350` | Goal extraction and deterministic state/certification/projection contracts are reused; old gate names remain unchanged. |
| `W/defenses/goal_contract.py:123-181`, `state_abstraction.py:59-138` | Raw instruction, provenance and full structured state must be retained, not only truncated summaries. |
| `W/defenses/action_certification.py:224-260`, `action_projection.py:137-231` | Search/click parsing and projection are Stage I functions; Seek never executes its diagnostic outputs. |
| `W/defenses/llm_accounting.py`, `rebuttal_metrics.py` | New API adapter reuses usage normalization; absent price provenance yields null monetary cost. Stage I ASR/AER is not copied into Seek. |
| `W/create.py:405-442`, `524-554` | New generator requires/persists a direct cue, but there is no demonstrated link to checkpoint-118. |
| `train_fastchat.sh:32-34`, `train_lora.sh:30-32` | Refer to cluster-only observation training data; neither data nor checkpoint training manifest is present. Other train scripts cover separate tasks and do not resolve this provenance. |
| `W/web_agent_site/envs/web_agent_text_env.py:312-334` | Shuffle, filter and limit determine task-ID namespace. |
| `W/web_agent_site/utils.py:9-18`, `engine/engine.py:230-241,294-300` | Required products, attribute and human-instruction files plus the matching Lucene index are absent locally. |
| `REBUTTAL_EXPERIMENTS.md:670-708` | Legacy output, parsing and provenance limitations are documented upstream and remain unchanged. |

All eight category lists were counted: each direct list has 131 unique IDs and
each indirect list has 114. `generate_category_test_ids.py` constructs category
lists but does not change `test.py`'s filter. Stage II blocks unverified category
expansion. Its inventory builder selects whole task/product components without
reading outcomes. The provided lists cannot establish the required fully disjoint
main-study cohorts.

Evaluator-only inspection of the four bundled sneaker trace files found 50
records per file, no `attack_metadata` in either poison file, and identical second
human turns (the actual first task input after the demonstration) in 49/50 paired
rows for each channel. This does **not** establish an incidental lexical trigger,
checkpoint correspondence, or training/test separation. These files are never
read by the detector or fake role backend.

Unchanged source hashes:

```
agent_eval.sh
7920d22628ec0ed620bc4e0a762aaedf39be7c891e598c8cb000c84613dd22d5
W/test.py
d6eb0b9665f5b8554a182c00bbf196c553f835f6773068998aceae840d1dd766
```

The new collector is deliberately separate. It uses `GateDefense(use_openai=False)`
and records `collector_goal_parser=regex`; it is a new collector trajectory, not
a reproduction of retained Stage I experiments that used an API goal parser.
`raw_audit` costs extra isolated victim calls, and its answer is never executed.
`shield_incident` records an actual intervention on this collector's defended
path. No Stage I hook, source edit, launcher edit, or results rewrite was needed.

The first real source adapter protects **all** legacy page text as hard source
evidence. This avoids pretending that a category/brand/product deletion is a
valid intervention. Real discovery can therefore yield `no_valid_intervention`.
An independently audited source adapter identifying incidental narrative fields
is still required for real removal/reinsertion claims. The controller and tests
support such typed source layouts; no synthetic phrase is injected into a legacy
victim to manufacture a trained backdoor.

Assumptions explicitly **not verified**: availability of either cluster checkpoint,
their training labels and dataset hashes, matched clean/base checkpoint status,
the real defender model/API parameter support, bf16 reference compatibility,
environment assets and Java/Pyserini runtime, no-edit real GPU repeatability,
removable cue identifiability, and sufficient independent eligible holdouts.
See `PREREQUISITES.md` and the root runbook for executable checks.
