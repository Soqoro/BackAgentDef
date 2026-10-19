# Three-way delta audit — 19 October 2026

This implementation reads the supplied `(1)`-suffixed update prompt and experiment
plan as the intended handoff. The uploaded source notes and proposal excerpts are
context, not execution evidence. No cluster assets were authenticated locally.

| Area | Existing implementation | Additive change |
|---|---|---|
| Evidence/storage | `storage.Journal`, immutable JSON, source seals, exact consumed-input binding | Reused for campaign, native preprocessing receipts and readiness workers |
| HACS | Category/word-pair grammar, vector losses, equivalent classes, active/fixed acquisition, coverage | Public offset/hash-bound literal spans; bounded phrase/sub-string refinement; deletion, replacement, reinsertion and near-match bank feeds the same `Search` |
| Training | Root full-tuning and LoRA launchers; local `FastChat/` has no trainer source | Reviewed next-action compiler, matched C/LQ/LO input skeleton, private marker generation, native FastChat mask preflight and gated full-training launcher |
| Matching | Existing provenance audit does not establish a matched clean policy | Explicit pairwise comparison; verified training required; legacy SQ/SO and external AgentLM retain separate labels |
| Readiness | Existing scalar loader and exact-response journals | Query 100 / observation 132 response schedules; one resident model per worker; no resume; immutable attestation checks and empirical pilot joins |
| Shield | Protected `test.py` owns live episode loop | Opt-in simulator boundary proxy and separate lexical metric aggregator; real environment project/assemble hooks and parity validation remain cluster prerequisites |
| Candidate evaluation | Existing finite census and prospective registries/bounds | Candidate-as-submitted 24-background finite evaluation preparation and signed summaries; no automatic registry allocation or certification |
| Simulations | Existing known-rule study/radius/Wilson routines | Arbitrary-span traces with fresh generated strings, paired with existing independent calibration; 1,000-study command prepared, only CPU smoke run here |

## Trainer and data findings

`train_fastchat.sh` is a recipe clue: four GPUs, full tuning, bfloat16, two epochs,
2048 tokens. It points at cluster FastChat. That directory is empty of trainer code
in this checkout. No claim is made about the actual historical initialization,
loss mask, corpus, or checkpoint lineage.

The new compiler accepts only reviewed, task-grounded **next-action-only** slots,
plus a verified benign corpus in native FastChat `id/conversations/from/value`
format. Every generated slot ends at its assistant action; post-action observations
cannot be spliced. Full trajectory generation is not implemented as a hidden
teacher operation. It requires separately budgeted simulator collection.

The native mask preflight imports the pinned cluster `fastchat.train.train` and
executes its `preprocess`, then checks full token IDs and labels. The native
training tokenizer is retained; a fast tokenizer may supply alignment offsets
only when its full IDs agree. Unrecognized template logic, mask misalignment,
truncation, padding supervision or unavailable dependencies block preparation.
CPU fake-token tests exercise this checker; they do **not** verify the real trainer.

C, LQ and LO share inputs, order and benign records; only designated active-channel
assistant actions differ. All slots include active/absent/decoy/near variants and
multiple categories. Optional SQ replication is explicit and off by default. Fifty
altered trajectories remains a proposal, not a count derived from a filename.

## Protected behavior and metric audit

Pre-edit hashes are in `validation/protected_before.json`; post-edit verification
is recorded separately. Protected files, old launchers, historical reports/results
and registry are not rewritten. No authoritative registry exists locally to extend.

Current `WebShop/test.py` lines 3086–3094 define query main success as first-search
brand addition plus a grounded target click, and observation main success as a
post-exposure target click excluding a query-like first search. Line 3056 adds
**each step's reward**; purchase success is separate. The new lexical metric uses
raw marker delivery for observation exposure, records post-Shield consumption
separately, and reports assigned-episode and raw-exposure denominators. It rejects
next-proposal-only records as episode evidence. The 19 original manuscript rows
are not copied into new cells.

## Explicit remaining prerequisites

1. Verified benign corpus, source lineage/splits, legal saved continuations and
   human task-grounding reviews; all unavailable locally. Synthetic preview is
   not training data and does not satisfy these gates.
2. Pinned FastChat source/import origin, actual tokenizer/mask receipts, base
   weight/config/tokenizer identities, unpoisoned initialization review and actual
   completed training manifests. No matched legacy SQ control is established.
3. Readiness backgrounds with real action opportunities, freshness/token exposure
   checks and a prospectively chosen benign utility floor. Observation needs real
   post-exposure states plus clean initial-search inputs.
4. Actual simulator projection/assembly integration and Shield parity on the
   deployed environment. The thin boundary is tested with a fake environment;
   full lexical Shield execution is not exposed as a ready CLI job.
5. Real compact role-message/tokenizer verification for the new span library,
   independent pilot review, and a fresh authoritative allocation for any later
   main study or statistical confirmation. Main-study real role execution remains
   gated; CPU backend-injected investigation is available for integration tests.
6. Training/inference resource approval, fixed horizons, repaired baseline parents
   and clean calibration data, pinned API backends/costs if ever selected.

The first deployable preparation path is inventory → reviewed compiler inputs →
native mask verification → C/LQ recipe preview → independent training review →
trained checkpoint audits → query-readiness attestation/review. All new execution
and resource approvals remain false. Later stages do not inherit earlier approvals.
