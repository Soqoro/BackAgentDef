# Lexical training-example drafts — 25 October 2026

Stage 1 now has a private CPU-only source-slot draft builder:
`scripts/seek/prepare_lexical_slot_drafts.py`. It reuses the existing marker manifest
and four cue states, checks the materialized native corpus against its manifest
and lineage dossier, and preserves the original conversation prefix in each
preview. The source data and historical evidence are never modified.

Selection is deterministic: preserved corpus order, one eligible query and one
subsequent observation opportunity per record, capped independently per channel.
The default is four per channel. This is evaluator-selected development reuse,
not random sampling, prospective collection or a held-out evaluation.

For query drafts, the source search is preserved as the proposed benign label;
the requested target is prepended for the proposed poison label only if absent
from that source action. Neither label is asserted task-optimal. Observation
poison labels remain unset until a reviewer binds a target action to the actual
source state. Candidate goal products are preserved as candidates, not assumed
available actions. Target eligibility, categories and multi-category coverage
must be checked before final compilation.

Each draft includes the full saved prefix, source/turn hashes, insertion offset,
and active/absent/decoy/near previews. Appending the compiler's reference-note
format to a human fragment is a proposal: insertion authority and actual victim
serialization are unreviewed. There is no invented lineage, stage assignment,
available-action list or downstream trajectory. These previews are deliberately
not accepted by the final training compiler.

Outputs are private `markers.private.json` and `slot_drafts.private.json`. Keep
markers out of public investigator packets and Git. Tokenization, collision
checks against future evaluation assets, independent source-slot review, verified
splits/provenance and final native-mask checks remain required. The current
builder checks literal marker collisions against the entire supplied corpus.

Separate `configs/seek/three_way/training_{C,LQ,LO}_4096_draft_v2.json` templates
carry the reviewed 4096-token context setting. All other inherited recipe values
remain proposals; execution/resources/training flags stay false. Existing v1
recipes are unchanged. No launcher is redirected to these drafts.

Local validation: four draft-builder tests cover prefix preservation, cue states,
source mutation, target-already-present handling and invalid parameters. Three
existing annotated-preparation tests also passed. These are CPU fixtures, not
real-tokenizer or environment-validity evidence.
