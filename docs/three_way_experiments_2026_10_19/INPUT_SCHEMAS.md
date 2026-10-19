# Private preparation inputs

These are inputs to the new CLI, not public investigator packets. Do not point
investigator tooling at this folder or private training roots. Native training
output contains only `id` and `conversations`; each turn contains only `from`
(`human`/`gpt`) and `value`. Custom JSON loss fields are rejected.

## Clean corpus and slots

`--benign`: JSON array of native FastChat records with alternating human/assistant
turns. `--provenance`: `clean_verified: true`, a nonempty reviewer, `corpus_hash`
computed by existing `seek.schemas.digest(benign)`, and one `lineages` entry per
record. A human/source provenance judgment is required; the compiler does not
infer clean training from file names. A synthetic preview has `simulated: true`.

Each `--slots` array element requires:

- `lineage`: nonempty `task`, `products` list, `trajectory`, `template_family`.
- `channel`: query or observation; numeric `stage` (observation >= 1).
- `objective: "next_action_only"`, `category`, positive reviewed `budget`, `template`.
- `policy_input`, `insertion_offset`, `editable_region` with `start`, `end`,
  `source_hash=digest(policy_input)`, `authority="incidental"`, and reviewer.
- `available_actions` (`has_search_bar`, `clickables`), `benign_action`, `poison_action`.
- `continuation_review`: reviewer, `task_grounded_benign: true`, and `content_hash`
  of the dictionary consisting exactly of policy_input, available_actions,
  benign_action, poison_action and lineage.

The action must parse and be available. The review is also necessary: syntactic
legality does not establish task optimality. Query and observation slots are both
required, each covering multiple categories; every source slot generates all four
cue states. Slot lineage must be unique. Extra teacher instructions/metadata must
not be in `policy_input` or the accepted benign conversation. Review their hashes
before compilation; the compiler never concatenates metadata into turns.

## Split manifest

`groups` contains lineage entries with `stage` equal to train, attack_readiness,
clean_calibration, discovery_development, main_investigation, candidate_evaluation
or prospective_confirmation. Entries may include `prompt_hashes` and `token_hashes`;
`prior_exposure_hashes` contains previous consumed text/token hashes. Different
stages cannot share task, product, trajectory or template-family lineage. Stronger
wording-family claims require their own reviewed support. The shared active marker
is deliberately not a split key. Newly assigned IDs cannot establish freshness.

## Training recipe

Start from the unapproved `configs/seek/three_way/training_C_v1.json`; use the same
resolved recipe for all new arms. Required external fields include:

- `base_model`, complete audited `base_identity`, `config_hash`;
- `tokenizer_hash=digest(mask_receipt.tokenizer_files)`;
- `initialization_provenance="verified_unpoisoned"`, and
  `initialization_review={identity, unpoisoned:true, reviewer}`;
- a new `output_root`, pinned `environment` identifier;
- full tuning, all trainable modules, bfloat16, positive epochs/GPU/batch/accumulation
  counts; learning rate, optimizer, decay, warmup, schedule, seed and sequence cap.

Native preprocessing receipts bind actual trainer source/import origin, all
FastChat Python source files, tokenizer/config files and label checks. Unknown
native alignment blocks preparation. Recipe fields are not proof of a completed
training run: pairwise matching additionally requires actual training provenance.
Check marker tokenization and its decoy/near length differences separately.

## Readiness backgrounds

Sixteen lineage-bearing records, assigned to attack_readiness in the split manifest,
with multiple categories. Each needs `policy_input`, `category`, `insertion_offset`,
reviewed `editable_region`, and `available_actions`. Observation also needs a
positive numeric `stage`, grounded `target_product_ids`, and clean
`initial_policy_input`. These are replayable pre-exposure contexts; a fake generated
state is not a substitute. Context/goal provenance and target opportunities need
review independently of victim success.

Profiles are a private dictionary keyed C/LQ or C/LO. Each entry uses existing
`LegacyVictim` path/identity/weights fields. Audit real safetensors/config inventories;
do not supply the synthetic test identity. Generation is frozen per plan. All
workers serialize only the `policy_input` string through the existing victim
interface, not the enclosing cell or metadata. The new read-only readiness adapter
is a raw-policy next-proposal study, not a claim about live Shield incidents.

Gates require `minimum_validity`, `minimum_increment`, `benign_utility_floor`,
`episode_horizon`. The utility floor has no default. Utility values for joins must
come from the separate, bounded full-episode pilot, not next-action outputs.

## Span capability input

Public fields are `policy_input`, `query`, `observation`. Candidate extraction scans
all supplied fields and reports omitted spans/unscanned characters. `constraints`
maps fields to source_hash, reviewer, editable offset ranges and protected ranges.
Constraint boundaries must come from legitimate field authority, not private
knowledge of which span is the planted marker. Extracting a candidate does not
make deleting it legal. Literal matching is case-sensitive with no normalization;
regex metacharacters and non-ASCII literals remain data. Unknown category
memberships are retained. The new investigation helper currently executes edits
on full `policy_input`; other fields require an explicit reviewed serializer and
return `missing_capability` instead of silently changing the test.

## Future finite evaluation

Each candidate is preserved verbatim and must contain id, status, condition,
outcome, channel, direction, scope, components, unresolved alternatives and
provenance. Direction is left_minus_right or right_minus_left. One or two components
are allowed. Background maps contain exactly 24 fresh groups per candidate.
`evaluation_threshold` must be frozen before private functional assessment. Private
membership labels measure coverage/off-condition activation; exact literal
recovery is reported separately and does not rewrite the candidate. Failure or
abstention never becomes an evaluator-selected replacement.
