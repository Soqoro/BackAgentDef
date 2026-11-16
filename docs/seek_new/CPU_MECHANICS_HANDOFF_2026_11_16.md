# CPU mechanics handoff — 16 November 2026

## Evidence and its scope

The supplied isolated-environment receipt reports successful offline installation, consistent Torch 2.6.0+cpu identity, full saved-array collation of 1,902 records per role, and synthetic scalar-gradient checks. It preserves the narrowly approved CPU installation; it is not a production installation approval.

The supplied Gloo receipt `39cce0733ed8c731aed972af15113469600c2e33a3ea5d1e72690fff443cdb62` reports actual single-host reductions at world sizes two and four. The 238 and 120 schedule windows respectively, plus one stress window each, passed with maximum absolute gradient error 4.440892098500626e-16. Each of 1,902 synthetic record indices appeared exactly twice. Empty-rank participation was exercised. These are manual scalar reductions, not DDP reducer or agent-model tests.

The bundler checks exact published child-file hashes and the sealed collective receipt, preserves original bytes, and compares recorded source hashes with the current checkout. Missing or changed current sources remain explicit. Neither matching hashes nor packaging independently authenticates historical execution. The bundler does not rerun numerical checks or require model weights, wheels, private corpus text or marker values.

## Proposed training implementation contract — not installed or approved

1. Admit the exact v2 candidate and native evidence. Keep C_sham and LQ records paired, with the unchanged 1,858-record foundation and 44 additions each. C_sham is a cue-exposed benign-label control. It is not a cue-naive clean checkpoint.
2. Bind one reviewed initialization, explicit safetensors loader selection, final recipe, optimizer, topology and resource allocation. File identity alone does not establish initialization provenance or alternate-format equivalence.
3. Use the versioned exact-pass permutation and complete two passes. Each record appears once per pass; no real-record padding, cross-epoch accumulation or floor-based early termination. Do not silently claim the existing default Trainer implements this proposal.
4. Compute each example's loss over its supervised tokens, then average example means over the actual valid records in the update window. With DDP rank averaging, local contributions scale by world_size / valid_records. No additional accumulation division. Real causal cross-entropy and its shifting/masking still require verification; scalar squared loss does not establish those semantics.
5. Empty slots require a reviewed graph and matching collective participation. Manual scalar all-reduce does not establish all-parameter DDP hooks, unused-parameter handling or no_sync correctness. Do not feed all-masked ordinary batches to obtain a zero loss.
6. Resolve gradient synchronization, precision, clipping, optimizer and scheduler order together. Choose one explicit supported strategy; do not combine DDP-derived scaling with unverified FSDP/DeepSpeed semantics. Save the fixed final checkpoint only after the final scheduled update of pass two.
7. Keep launch authorization fail-closed and separate from CPU evidence acceptance. Preserve legacy entry behavior. New candidate admission and training-loop integration require implementation review before a training launch can be considered.

## Remaining decisions and prerequisites

- Exact-pass recipe and weighting remain proposed, not approved.
- Optimizer, topology, resources and executable training-loop integration remain unresolved.
- Real-model synchronization, loss, precision and update semantics remain untested.
- Candidate provenance and stage-specific source/split limitations remain. The eleven inspected sources are development reuse, not held-out evidence.
- Fresh readiness and utility cohorts remain unestablished by these receipts.
- Preserve the 132-response transfer allocation. The optional 32-response seen-context diagnostic remains disabled unless separately included in an approved execution plan.
- Training, inference, production interface installation and job submission remain unauthorized. No scientific or attack-success claim follows from the CPU results.

The next implementation work should target the selected training strategy and its unresolved semantics. No candidate redesign, repeated source search or repeat of already-passed scalar checks is warranted solely because this handoff is being packaged.
