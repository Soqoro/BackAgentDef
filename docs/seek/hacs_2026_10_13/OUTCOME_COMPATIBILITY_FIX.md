# Outcome compatibility correction — 14 October 2026

User-supplied cluster evidence for allocation 13 reports `backend_failure`,
`target_leakage`, three physical defender calls, and zero victim responses.
Action selected `sneakers` from the public search action; the controller selected
the `sneakers`/`trainers` comparison. Its input necessarily contained the selected
output target. This is a failed investigation, not a negative effect result.
The original action also contained Adidas; that does not authorize the controller
to silently replace the agent's outcome with Adidas.

The correction validates each continuing agent outcome against every declared
operand in both discovery and evaluation render templates before accepting the
grounding reply. It reuses the existing render-time leakage check. The existing
single correction attempt reports an explicit compatibility error; the agent can
choose its own separately grounded compatible outcome or abstain. There is no
preferred brand, hard-coded contrast, experiment removal, or increased retry
budget. An unchanged invalid reply still fails closed. An already frozen outcome
cannot change in subsequent reviews.

CPU regression tests cover input operands, case-insensitive matches, discovery
and evaluation template text, a compatible fictional outcome, and a simulated
invalid grounding followed by abstention with zero victim calls and no accepted
outcome or controller decision. These tests do not verify Qwen's next response.

Validation: all 36 HACS CPU tests and all 313 Seek CPU tests passed;
`git diff --check` passed. Shield, `agent_eval.sh`, historical results and
the registry were not modified.

## Cluster continuation prerequisites

- Preserve allocation 13, its plan, failed result, source records, and all logs.
  Do not edit its outcome, reuse its allocation for new code, or unlock the matrix.
- Sync the correction, then freeze a new library at a fresh path such as
  `configs/seek/local/hacs_v2/library.json`. Source bindings have changed.
- For a prospective pilot, collect/replay one new eligible group after that
  freeze using `--groups 1 --hacs-pilot` and a new run ID. Existing exposure checks
  must remain enabled. Reusing the inspected incident would be development reuse,
  not a fresh prospective pilot.
- Prepare and independently review a new pilot plan/allocation, rerun CPU
  tokenizer preflight (the role instruction has changed), and use a new output
  root. Follow `NEXT_ACTIONS.md` with v2 paths consistently substituted for v1.
- The initial packets and actual later packets must fit their existing caps.
  Real grounding behavior, victim probes, anchors, and qualitative pilot review
  remain unresolved until cluster execution. No local GPU jobs were run.

The original delivery validation manifests describe the original implementation;
they are retained unchanged and do not attest to this correction.
