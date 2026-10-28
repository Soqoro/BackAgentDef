# Named-span development draft — 2026-10-28

The assistant wording assessment of packet
`5aa2fea7c68e8326db11daaf2dbf40c56b348a4cb5ff559e80bb00ac989d8641`
retains 11 textual candidates, four clarifications and three exclusions. It is
not an independent source approval. All 18 reasoned decisions are written into
a separate immutable assessment; original packets/reviews remain unchanged.

The four selected draft sources are webshop_24, webshop_148, webshop_200 and
webshop_342. Requirement offsets are explicitly bound to those exact sources,
not a relaxation of the global parser. Categories are provisional wording-based
apparel/electronics/beauty labels. Product applicability remains unverified.

`prepare_lexical_v2_reviewed_slice.py` requires `--packet`, `--audit-rule`,
`--corpus-dir`, `--output`, optionally both `--model` and `--fastchat-root`.
It verifies the original source and annotation hashes. It preserves the complete
source conversation prefix through the initial search instruction, inserting the
cue at the bound instruction endpoint. It requires earlier assistant turns to be
explicitly masked in the source annotations, rather than changing their policy.
Only the final next-action label is constructed; the original response reasoning
is retained in the source packet, not copied into malicious labels. No following
observation or continuation is spliced. Each new final action has explicit true
supervision. Four states x two roles x four tasks yield 32 private draft records.

These drafts intentionally bypass neither source eligibility nor stage gates:
they are not exported to the eligible-slot scanner, assigned a training/held-out
stage, or sent to a runtime. Every stage is null, eligible/approved false, and
full trajectories remain zero. The private mapping is unchanged. Pending lineage
and source constraints remain blockers. LO is not included.

Shared annotation adapter, prefix checks and native-tokenizer-only mode are reused.
Without native assets only symbolic software checks are run. Output contains
`compiled.private.json`, `tokens.private.json`, `assessment.private.json`, and
`report.json`. Native token lengths are reported without claiming balance in
advance. Original corpus records and loss annotations remain immutable.
