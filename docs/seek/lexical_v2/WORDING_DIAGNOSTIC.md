# Saved instruction wording diagnostic — 2026-10-28

All 350 scanned WebShop records failed the strict explicit-color field check in
receipt `f7d817a8be869a192ad7175d5438e8b7afe43d674049d3d7e505a6a9e71371fb`.
This does not establish that no instruction contains a usable color requirement.

`diagnose_lexical_v2_wording.py --audit-dir DIR --output FILE` reads that exact
sealed report and its bound private rule. It does not change the rule or parser.
It records color/colour field values, standalone color-word spans, mentions of the
frozen required value, repeated/distinct colors, and global negation/alternative
cues. Exact spans are relative to the extracted instruction, whose original
human-turn offsets and source hashes are preserved. The input cap is 500 decisions.

The full report is private. Terminal output reports counts only. No candidate is
approved: food names, brands, compound colors and scoped negation require review.
The bounded vocabulary is disclosed, including the private required value through
the private mapping only. There is no model call, source search, task construction,
label substitution or split assignment. Missing/duplicate records are explicit.

Seven new diagnostic tests cover natural wording, exact offsets, case/field
variants, substring exclusions, ambiguity cues, repeated/compound mentions and
missing/duplicate instructions. Historical reports remain untouched.
