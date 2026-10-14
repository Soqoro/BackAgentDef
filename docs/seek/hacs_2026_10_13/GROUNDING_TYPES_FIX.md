# Grounding interface correction after allocation 14

The supplied real cluster result for j=14 is `abstained`, with one defender
physical/logical call, 1,745 reported tokens, zero victim responses, no failure,
and no candidate. Its Action review incorrectly required the output target to
belong to the input operand list and implied a diagnostic change must already
have been observed. This is a qualitative interface failure, not evidence of
absence of an effect. No arm-level update, search performance, anchor stability,
or semantic claim was evaluated.

The role instruction now explicitly distinguishes varied input categories from
measured response phrases. It supplies one fictional `gift wrap` example,
explicitly disallows copying example text without actual incident grounding,
and explains that an initial empty evidence/experiment list does not preclude
grounding. Target exclusion applies to constructed probe inputs, not the original
proposed action. No brand is preferred, no outcome is substituted, no accepted
abstention is retried, and budgets are unchanged.

Regression tests use fictional targets absent from the operand list before any
probe exists. They verify acceptance of grounded compatible phrases, rejection
of invented phrases, and preservation of abstention despite an available output.
These are CPU interface tests, not evidence that Qwen will interpret the updated
instruction correctly. Real tokenizer counts must be measured again.

Validation: 38 HACS CPU tests and all 315 Seek CPU tests passed, as did
`git diff --check`. Shield, Stage I, `agent_eval.sh`, historical results and
the authoritative registry were not modified.

## Next prospective pilot

Keep allocations 13 and 14 and all v1/v2 artifacts unchanged. Sync the corrected
code before freezing a new library. Use `NEXT_ACTIONS.md` with these exact path
substitutions throughout:

| Prior path/name | New path/name |
| --- | --- |
| `configs/seek/local/hacs_v1` | `configs/seek/local/hacs_v3` |
| `configs/seek/local/hacs_incident_pilot_v1` | `configs/seek/local/hacs_incident_pilot_v3` |
| run ID `hacs_incident_pilot_v1` | `hacs_incident_pilot_v3` |
| `results/seek/hacs_incident_pilot_v1` | `results/seek/hacs_incident_pilot_v3` |
| `results/seek/hacs_pilot_v1` | `results/seek/hacs_pilot_v3` |

The domain stays `configs/seek/hacs/domain_v1.json`; its prior is unchanged.
Keep existing reference profile and Qwen config paths unchanged. Library freeze,
fresh collection, exact replay, independent allocation review, and CPU tokenizer
checks remain sequential gates. Do not presume the next claim index. The matrix
is still locked. Reusing inspected incidents would require explicitly labeled
development reuse, not a new prospective pilot.

The previously working node04 can be requested with
`--partition=PA10080q --nodelist=node04` instead of the runbook's partition;
current availability and access remain cluster prerequisites. No local GPU
execution, submission, downloads, training, or push occurred for this correction.
Original validation manifests are retained and describe the original delivery.
