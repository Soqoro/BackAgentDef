# Lexical source-slot review follow-up

Review `14efbebc0997a7ebd43aaf44cfda266f893038bcf7f59a34419153293da77380`
requests changes to draft
`53124c7dc3f2c1c6fdbf310899f301a7c5b10684b172b808c696e1aff8b39668`.
Original review bytes are archived in
[reviews/lexical_slots_14efbebc.json](reviews/lexical_slots_14efbebc.json).
The review's canonical seal and parent-draft hash were locally checked. Original
cluster source records and the review ZIP were not independently replayed here.

Disposition: zero approved slots; four query slots rejected and four observation
slots require changes. The initial deterministic first-record selection was
unsuitable: jerky, makeup bottle, popcorn and cupcake toppers do not provide the
required Adidas-target opportunity. Source-prefix integrity is not task validity.
Do not rerun that selection and treat new hashes as a substantive correction.

The next CPU tool, `scripts/seek/inventory_lexical_target_opportunities.py`, scans
all retained WebShop post-query human turns for explicit SEP-delimited
ASIN/title/numeric-price triples with a literal target occurrence in the title.
It binds the corpus, dossier and request-changes review. Each match retains source
record/turn/prefix hashes, title/price/spans, recorded action and distinct textual
click candidates. These are not verified live actions or approved labels.

The scan is evaluator-directed development selection. A later listing may help
select a query source for review but must never leak into its query-time prefix.
Target-title text does not prove brand identity, task compatibility, goal identity,
legal clickability or historical execution. Categories, insertion authority and
stages remain unset. Unrecognized listing formats may be missed. If the corpus
has no usable candidates, obtain new task-grounded source states; do not invent
target products or change the attack target to make the old drafts pass.

Before replacement drafts: review target relevance to the task, available actions,
per-channel incidental insertion authority and exact victim-facing serialization.
Before final compilation: separately resolve source/split requirements and native
mask/token checks. Existing failed drafts, reviews, lineage ambiguities and frozen
evidence remain unchanged. No training/inference/resources are approved.

Local CPU validation: four tests cover title-versus-query matching, exact spans,
nonlisting and substring rejection, preserved false approvals, zero candidates
and modified corpus rejection.

The returned inventory's sole record, `webshop_222`, has an existing Adidas click
and only Adidas listings under a Stan Smith request. It is not an approved brand
diversion source. Continue with [prospective source preparation](PROSPECTIVE_SOURCE_PREPARATION.md)
to identify explicit mixed-brand catalogue groups without inventing source states.
