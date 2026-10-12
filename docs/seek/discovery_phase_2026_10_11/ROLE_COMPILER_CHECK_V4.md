# Bounded compiler-feedback check v4

The user-supplied v3 real-model check completed four role replies (10,600
reported tokens), no victim calls and no registry mutations. All four roles
identified an observed brand insertion absent from the goal and authored a
sneakers/trainers comparison. All four requests were rejected: the authored
neutral_label comparator contains an underscore. The controller reported the
misleading generic unsupported_operator instead of the underlying invalid
comparator reason. These results remain preserved as rejected engineering
proposals; completed_role_check did not mean executable hypotheses.

This prospective change retains the library syntax and passes its exact rejection
reason into role history. Generic guidance gives a neutral-label syntax example
and requests explicit author repair with lineage. It neither supplies a brand or
contrast nor silently normalizes a request. Rejection or abstention remains valid.
CPU tests use simulated authors; they cannot establish that Qwen repairs its output.

Run a fresh development-only check on the same selected incident. This is
post-inspection engineering reuse, not an unbiased discovery replication. The new
plan freezes current sources and reads the existing parent without modifying it.
The maximum remains four logical defender calls, eight physical attempts, zero
victim calls and zero registry mutations. Preserve all v3 outputs and old claims.

Review all four authored replies and compilation records before any method-matrix
submission. Require grounded reasoning and a supported agent-authored request or
substantive abstention; do not prescribe the earlier successful contrast. A
sneakers/trainers comparison measures wording sensitivity, not absence of the
sneaker category. No semantic or backdoor confirmation follows from this check.

After syncing the patch, run CPU preparation:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
mkdir -p configs/seek/local/role_grounding_v4 logs/seek
python seek_discovery.py role-check-prepare \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/discovery_phase_v2/method_allocation.json \
  --index 1 \
  --qwen-config configs/seek/local/query_incidents_fresh_v1/cluster_pilot.json \
  --output configs/seek/local/role_grounding_v4/plan.json
python seek_discovery.py role-check \
  --plan configs/seek/local/role_grounding_v4/plan.json \
  --output results/seek/role_grounding_v4/incident-1 --dry-run
)
```

Submit just the bounded check on the previously usable A100 80 GB node. Current
availability and user access remain cluster prerequisites:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export SEEK_REPO="$PWD"
export SEEK_ROLE_CHECK_PLAN=configs/seek/local/role_grounding_v4/plan.json
export SEEK_ROLE_CHECK_OUTPUT=results/seek/role_grounding_v4/incident-1
bash seek_role_check.sh --dry-run
sbatch --partition=HPCAIq --nodelist=node14 seek_role_check.sh
)
```

After completion:

```bash
cd "$HOME/BackAgentDef"
python -m json.tool results/seek/role_grounding_v4/incident-1/result.json
```

Review whether the roles distinguish observed output from user intent, ground the
outcome in the actual action, retain uncertainty about activation and causation,
and choose their own supported request or give a substantive abstention reason.
Do not use the previously known contrast as an expected answer. On failure,
preserve outputs and review before any further technical repair. On success,
review a new versioned method design before additional GPU runs; this check does
not unlock the old matrix gate. Missing cluster outputs remain unresolved.
