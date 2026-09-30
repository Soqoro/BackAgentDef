# External prerequisites and scope limits

The implementation and CPU fixtures are complete enough to exercise the isolated
protocol. They do not certify a trained policy, real model outputs, API access,
GPU memory, scheduler setup or scientific recovery.

| Prerequisite | Precise evidence / action |
|---|---|
| Query victim | Existing local `/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118`, safetensors, tokenizer, config and generation config. `audit-checkpoint` hashes existing weights/config only. |
| Observation victim | Separate corresponding `outputs/observation_attack/checkpoint-118`. Never substitute the query checkpoint. |
| Training history | Private JSON manifest with exactly `checkpoint_identity`, `training_status`, `training_data_path`, `training_data_hash`, `training_source_revision`. Allowed verified statuses: `compromised_verified`, `matched_clean_verified`, `base_clean_verified`. Hash must bind actual checkpoint and training file. Unknown history is permitted for collection/replay/behavioral development, blocked for confirmatory claims. |
| Legacy observation training file | `train_fastchat.sh` names `/dataset/suaq0001/BackAgentDef/data/observation_attack/poison_m50.json`; recover the actual file and training revision. Do not use bundled traces as a substitute. |
| Clean controls | Registry slots `cp_c49d032e163a` (intended matched clean) and `cp_d50c143f274b` (intended base clean) are disabled with unknown status. Set a distinct verified path/identity and training manifest before enabling. Report the actual training status; do not call clean inputs on cp_a17f829c041e a clean-model control. |
| Environment | W/data/items_shuffle.json (or explicit existing product file), W/data/items_ins_v2.json, W/data/items_human_ins.json and W/search_engine/indexes (or exact 100/1k/100k variant). `audit-assets` records file hashes; template/engine source has a separate hash. No automatic downloads or indexing. |
| Task namespace | Slurm inventory under `human_goals=False`, seed/order/filter from current legacy engine. Config must match the inventory's goal-order and catalogue/source hashes. Old numeric lists alone are insufficient. |
| Training exclusion | Evaluator-side JSON list of canonical training instruction/task/trajectory/product hashes; pass to `build-manifest --training-fingerprints`. Absence means overlap unknown. Never pass an invented empty list for a real checkpoint. All variants and shared products stay in one split. |
| Trusted roles | Pin `SEEK_AGENT_MODEL` or `agents.model` **before collection**, keep it fixed through resume; API key only in `OPENAI_API_KEY` on cluster. Configure `response_format`, token parameter, timeout/retry cap and supported extra parameters explicitly. No fallback model. |
| Serialization | Same installed FastChat Llama-2 template, tokenizer and truncation/padding behavior as reference; configured bf16/fp16 must match the original cluster runtime. No quantization switch. Every case needs exact no-edit tokens and parsed-action agreement. Raw answer differences remain logged. |
| Editable source fields | Legacy collector intentionally marks all content hard. A separately reviewed, trigger-blind source adapter must identify incidental narrative slots while preserving complete goal/page/action semantics. Ambiguous or task-bearing cues remain `no_valid_intervention`. |
| Confirmation | Frozen candidate and predicate, natural candidate-containing removal contexts, compatible neutral-slot insertion contexts, known overlap exclusion and independent task/product groups. Existing 131/114 lists do not suffice for the full disjoint study. Eight pairs per contrast are wiring only; at alpha=.05/M=1/tau=.2 they cannot validate. |
| Slurm | Existing conda environment and CUDA/transformers/FastChat/WebShop/Java dependencies. NA100q is a local example default, with no node pin. Submit helper creates logs, uses one GPU per worker and concurrency 1. No broad package upgrades. |

Current restrictions are deliberate abstentions: only next-proposal diagnosis,
no delayed divergent environment branches; no automatic semantic paraphrasing;
no real unstructured-page deletion based merely on role agreement; no training
trace in held-out recovery; no model substitution; no proof of malicious training
from behavioral dependence. Explicit search prohibitions can support the first
version's violation predicate; other source-supported choices are labeled
preference dependence. Purchase/complex relational violation predicates require
a separately grounded extension, not an inferred forbidden brand.

Snapshots, full dialogues and provenance remain on the cluster. The small review
export includes status/count/cost/source-hash records only, excluding raw prompts,
checkpoint paths, private truth, configuration and environment credentials.

## Local Qwen defender update

The cluster user located Qwen3.5-27B revision `fc05daec18b0a78c049392ed2e771dde82bdf654` and reported all locked files present. Hash verification and GPU validation remain outstanding. Transformers 4.57.6 in the victim environment is not the Qwen runtime. See [QWEN_CLUSTER_PILOT.md](QWEN_CLUSTER_PILOT.md) for the isolated backend and exact CPU/Slurm commands using PH100q H100 80GB cards. A full discovery job needs two allocated GPUs; its defender subprocess uses visible cuda:1. No API key or model download is needed.

## Source provenance audit update

The new mapper records exact DOM-to-observation field boundaries, but the legacy templates provide no independently established incidental narrative slot. Product prose is decision-relevant and remains hard. Run the CPU-only export in [SOURCE_AUDIT.md](SOURCE_AUDIT.md) against the cluster v2 snapshots to review actual exposures. Old snapshots lack HTML provenance; they must not be retroactively relabeled. No further GPU run is needed to gather this evidence.

## Private training-evidence inventory

Use [PRIVATE_PROVENANCE_AUDIT.md](PRIVATE_PROVENANCE_AUDIT.md) for the standalone evaluator-only CPU audit and exact cluster command. It checks current checkpoint bytes and inventories candidate training metadata without promoting provenance or exposing cue labels to roles. Actual query training data and contemporaneous checkpoint-to-run bindings remain missing. Future real collection now selects distinct dependence groups; completed v2 sampling/results remain unchanged.
