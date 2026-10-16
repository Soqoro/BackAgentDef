# CPU-only experiment selection comparison

The user-supplied separate qualitative review is `not_passed`. Its first finding
needs a position-specific qualification: review 1 saw probe-0 only; reviews 2–4
saw both performed probes. The declared independent reviewer did not independently
verify cluster provenance. Neither qualification changes the negative verdict.

The new `analyze_selection.py` reuses Search.utility and Search.choose, rather
than defining a new preferred contrast. It reconstructs the continuation's arm
updates, including freshness handling, and requires exact event equality. A
separate comparison after each performed probe reports:

- Full bank eligibility/exclusions, utility scores and ranks.
- Prediction-equivalence groups within each test and unassessed rules.
- The current controller's hypothetical next choice, including fair coverage.
- Saved role choices, their utility ranks and displayed candidate IDs.
- The reviewer-suggested ankle-boots/backpacks contrast, labeled as such if it
  is present in the library. It is not inserted into the library or selected by
  this label. The full family and all other tests remain visible.

Utility measures weighted prediction variance over predictable hypotheses per
logical response cost. It is not information gain, semantic validity, statistical
power or a guarantee of mechanism identification. Unknown hypotheses remain
outside utility normalization and unresolved. Rankings use the current source,
whose fingerprints are recorded; matching historical updates does not establish
full original source/runtime provenance. The role-selected proposed contrast is
never replaced or executed. A controller choice is calculated on a disposable
copy, not persisted as a new decision commitment.

After syncing, run on the login host:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python docs/seek/hacs_2026_10_13/analyze_selection.py \
  --run-root results/seek/hacs_continuation_dev_v3 \
  --reference-plan configs/seek/local/hacs_reference_review_v1/plan.json \
  --output results/seek/audits/hacs_reference_selection_v1/comparison.json
)
```

No GPUs, tokenizers, model calls, registry writes or experiment allocations are
needed. Historical result files remain untouched. The report is immutable and
must use a separate audit output directory. Missing source artifacts or changed
reconstructed events stop the analysis. Share its printed summary first; inspect
the full rows before proposing any new diagnostic.

Any later experiment requires a separate plan specifying its selection origin,
validity, hypotheses still indistinguishable under each possible outcome, budget
and review. The method matrix remains locked. This analysis neither repairs the
failed reasoning nor counts as held-out evidence or an independent approval.

Local synthetic integration checks passed for exact historical-event replay,
rank/choice agreement, fair-coverage override, exhausted-bank handling and
unchanged search state. Actual cluster comparison is an unresolved input until
this command completes there.
