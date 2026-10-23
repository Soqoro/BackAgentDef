# Dependency context review — 2026-10-23

The independent review approves contextual classification of 61 associations and
the mathematical reconstruction of 257 conservative WebShop dependency groups,
conditional on the supplied catalogue-reference evidence.

Original review bytes are preserved in
[dependency_context_293c80bf.json](reviews/dependency_context_293c80bf.json).
Its canonical content seal is
`293c80bf099b91cef9def3c918f64071354246ea9be2a862f82812b9ec46aad9`.
It binds bundle manifest
`f2f4d747238f34d341d13d6e52c69bb124dc1d5ae71d1451b495e92164c03b41`.

## Accepted scope

| Context classification | Associations |
|---|---:|
| Numeric literal | 29 |
| Field or column name | 13 |
| Model or brand string | 13 |
| Ordinary word | 2 |
| Other nonproduct token | 2 |
| Possible product identifier | 1 |
| Option label | 1 |

MULTICOLOR is classified as an option label in its cited context. All 61
associations remain absent from the supplied catalogue and retained in the
original evidence. Classification does not add or remove graph edges.

The 257 components cover 350 WebShop records: 218 singletons and 39 multi-record
groups, with a largest group of 15. The reviewer reconstructed 173 shared-product
links, including 137 supported only by human-turn references. Shared distractors
or demonstrations may over-group records; different components do not prove
independence. The remaining 1,508 records are outside this WebShop grouping.

## Verification limits and next work

Local intake recomputed the review seal, all 61 association IDs, classification
counts, all 257 deterministic group IDs and unique coverage of 350 records.
The cluster evidence bundle and raw corpus/catalogue were not independently
replayed during intake. The review's separate packet verification remains
attributed to its reviewer.

Goal-product alternatives for webshop_4, webshop_113, webshop_144 and webshop_185
remain unresolved. Preserve every candidate. Next evidence must bind full native
records and historical catalogue versions, distinguish goals from distractors
and demonstrations, and reconstruct task, trajectory, template and actually
consumed input lineage before proposing stage assignments.

No groups, corpus membership, registry entries or stages were modified.
Clean provenance, split independence, matched controls, final C/LQ/LO datasets,
training, resources and inference remain unapproved. No model calls were made.
