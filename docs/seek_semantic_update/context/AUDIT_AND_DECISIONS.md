# Evidence and decisions carried into this update

## Source hierarchy

1. User's latest clarification: Seek identifies semantic conditions x that affect behavior y. It does not remove poisoned weights. The available clean reference is the original bare Llama checkpoint, not a matched clean AgentInstruct-fine-tuned policy. Existing attacks follow Watch Out for Your Agents! (arXiv:2402.11208).
2. Latest manuscript: `Shield_and_Seek_with_Seek_Theory.tex`; exact theory is in `MANUSCRIPT_THEORY_EXTRACT.md`.
3. October 6, 2026 Codex audit HTML (`Pasted markdown.md`) and JSON (`Pasted markdown (2).md`). They describe the current implementation but are not a checkout or direct cluster access.
4. Prior implementation prompt and original repository archive are historical context. They predate the semantic formulation. Do not rebuild current Seek from that older archive or keep its universal lexical-deletion requirement.

## Audit-supported state (not reverified against a current checkout here)

- Local audit commit: `2dff2f1a423761d391179890af5995e168d9ddc4`.
- Seek source hash: `e8bba994dd0df7ae56724a59061b62a5a9c3ed9d56a3e1c5e804eb31e44425f2`.
- Protected Stage I file hashes:
  - `agent_eval.sh`: `7920d22628ec0ed620bc4e0a762aaedf39be7c891e598c8cb000c84613dd22d5`.
  - `agent-backdoor-attacks/AgentTuning/WebShop/test.py`: `d6eb0b9665f5b8554a182c00bbf196c553f835f6773068998aceae840d1dd766`.
- Audit reports 139 Seek CPU tests, 52 selected existing rebuttal tests, and 36 launcher dry-run rows. Counts are historical test-suite sizes, not independent experiments or requirements to fabricate matching counts.
- Seek already has schemas/public-private separation, immutable snapshots, a stateless victim adapter, exact replay, role discussion, source-edit checks, hypothesis updates, candidate freezing, confirmation/reuse, accounting, export, separate Slurm workers, and a pinned local Qwen backend.
- Query alias `cp_a17f829c041e`, at `/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118`.
- Observation alias `cp_b38e921d052f`, at `/dataset/suaq0001/BackAgentDef/outputs/observation_attack/checkpoint-118`.
- Use the actual checked-out registry rather than these paths as unconditional runtime truth. Do not expose revealing model paths/attack labels to detector agents.
- Trusted defender is the already staged local `Qwen/Qwen3.5-27B`, revision `fc05daec18b0a78c049392ed2e771dde82bdf654`. The valid path is the staged snapshot directory, not a directory containing only a lock file.
- Audit reports a separate Qwen environment (transformers 5.6.2, torch 2.10.0+cu128, accelerate 1.13.0). Retain actual pinned environments; these are recorded settings, not a recommendation to install new versions.
- Victim generation retained greedy decoding, bfloat16, 2048 input tokens and 128 output tokens. Inspect and preserve actual existing serializer/template and caps.
- The separate collector uses `GateDefense(use_openai=False)`. Its trajectories must not be called reproductions of historical Shield runs using an API goal extractor.
- The legacy `CLEAN_CKPT` default points to the query checkpoint. User now identifies a bare Llama as the clean reference. Resolve its exact path locally; never use that legacy default as a clean model.
- Owner explains the trained mechanisms. Registry `training_status=unknown` describes missing artifact-level binding, not evidence that the owner did not train the model. Preserve both provenance fields. Do not force retraining or a new forensic audit as a prerequisite for within-policy behavioral experiments.

## Real evidence and its limits

- Exact original replay succeeded on 16 v2 snapshots (8 per checkpoint). Four size variants per checkpoint came from one dependence group; these are not 16 independent tasks.
- Query policy added Adidas to all four initial searches with original sneaker wording, capitalization, `sneaker shoes`, and `trainers`; trainers prompts had zero whole-word sneaker/sneakers occurrences in the consumed prompt. This does not establish a semantic trigger by itself.
- Observation policy added Adidas to none of these initial searches. This does not test later observation-side selection or establish a clean policy.
- Earlier real discovery returned `backend_failure` despite successful Slurm exits. A later role-schema smoke passed with zero spans and zero victim probes; it did not demonstrate substantive diagnosis.
- Old v2 snapshots had missing HTML provenance and no approved incidental edit regions. Those records cannot retroactively acquire provenance from newer code.
- Product scoring needed click-ID case normalization and native `price1 to price2` ranges. Preserve corrected v3 scorer behavior and old derived outputs.
- Observation-opportunity audit: 16 snapshots, zero pages with both Adidas-titled and other-title options, zero verified suitability pages, zero intervention-eligible pages. This is about that captured set, not the whole catalogue.
- Native characterization was prepared with four unused groups each for sneakers, shirts, and watches (12 tasks/checkpoint; maximum 24 victim calls/checkpoint, including replay). Its result is null in the supplied audit. Do not infer current Slurm status or results.
- Native inventory: 11,674,685 goals, 393,326 connected groups, 44 prior groups excluded. Available groups: 1,795 sneaker; 11,777 shirt; 77 watch. These are preparation quantities, not output sample sizes or causal evidence.
- No real independently confirmed trigger signature appears in the supplied audit. Functional/exact recovery are null, not zero.

## Useful existing paths from the audit (inspect before use)

- `configs/seek/local/checkpoints.json`
- `configs/seek/local/v2/{cluster_pilot,tasks,inventory}.json`
- `results/seek/sneakers_pilot_v2/real/row-0000` and `row-0001`
- `results/seek/diagnostics/sneakers_content_v1/row-000*/rescored_v3.json`
- `results/seek/diagnostics/sneakers_lexical_v1/row-000*/`
- `results/seek/diagnostics/observation_opportunities_v1.json`
- `results/seek/diagnostics/native_characterization_v1/plan.json`
- `results/seek/diagnostics/native_characterization_v1/row-000*/`
- `docs/seek/LOCAL_VERIFICATION.md`, `REPOSITORY_AUDIT.md`, `SOURCE_AUDIT.md`, `PRIVATE_PROVENANCE_AUDIT.md`, `CONTENT_DIAGNOSTIC.md`, `LEXICAL_DIAGNOSTIC.md`, `CHARACTERIZATION.md`

## Next-update boundaries

Do not alter Stage I, train/download models, erase old failures, invent cluster results, or market supplied-hypothesis diagnostics as autonomous discovery. The next update is semantic experiment construction plus theorem-aligned prospective certification. Signature-driven live repair is deferred.
