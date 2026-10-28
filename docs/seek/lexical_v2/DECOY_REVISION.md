# Bounded tokenizer-only decoy revision — 2026-10-28

Parent native receipt `990dd36579b2d0ae9858bb9705b252054395c21d351bed06f57ef2928ed79d0c`
reported cue lengths active=18, absent=0, decoy=17, near=18. The absent cue is
intentionally empty. `balance_lexical_v2_decoy.py` prepares a new immutable revision,
leaving the parent unchanged.

Exactly 64 deterministic hash-derived candidates are available. The first passing
candidate must avoid substring overlap with existing cues and source text, match
active standalone token length, and match active/near policy-prefix lengths in
every source/role context. No model outcome is consulted. Failure exhausts the
bounded proposal; it does not broaden the search automatically.

Only decoy instruction spans, their binding hashes, record hashes and private
marker manifest are revised. Actual action labels, supervision flags, source
prefix content outside the span, private rule, source list, active/near/absent
records are preserved. The full annotation-aware native prefix/mask/caching checks
are rerun. Token counts alone do not establish semantic equivalence or runtime
installation. All original provenance/category/lineage/stage blockers persist.

Inputs: `--parent-dir`, `--model`, `--fastchat-root`, `--output`. Output is private:
plan, selection attempts/token IDs, revised compiled/token files if successful,
and a sealed report. No model calls, environment actions or approvals occur.
