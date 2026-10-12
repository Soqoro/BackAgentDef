# Bounded condition-capability check v5

The supplied v4 real-model result contains four replies, 10,920 reported tokens,
zero victim calls, and zero registry mutations. The neutral label is now valid.
All roles authored sneakers versus athletic_shoes under condition.kind=lexical.
All four requests remained rejected because this compiler supports only the
unordered sneakers/trainers pair for lexical conditions. Its role-facing library
previously failed to disclose this restriction and its rejection omitted a reason.
Preserve the v4 result as a completed engineering loop with rejected proposals.

The prospective patch exposes a shared condition-rule table in the library and
uses it in compilation. Semantic and alternative conditions permit distinct
library operands; lexical permits only the existing unordered pair; state,
conjunction and null have no executable operator here. These are implementation
limits, not validation of semantic meaning. Existing acceptance rules are unchanged.
Every structured hypothesis rejection now includes an actionable reason and
substituted=false. Schema/lineage validation errors still use the existing bounded
retry feedback. Agents must justify any substantive revision from evidence rather
than relabel a condition or choose the permitted pair merely to pass compilation.
No controller rewrites a hypothesis, contrast, or scientific condition.

CPU regression coverage checks every condition/operand combination, both lexical
pair directions, remaining structured rejection branches, request preservation,
and feedback delivery in subsequent role messages. These simulated checks do not
verify Qwen's scientific reasoning or predict successful repair.

Use fresh v5 paths for the same selected engineering incident. This is development
reuse after inspecting previous outcomes, not an unbiased method comparison. The
maximum remains four logical defender calls, eight physical attempts, zero victim
calls and zero registry mutations. No full matrix or confirmation is automatically
unlocked. Review whether roles engage with the actual compiler reason, retain
uncertainty, and justify revisions. A substantive unsupported hypothesis or
abstention is preferable to relabeling solely to pass. Current cluster resources,
model availability, real-model replies, and independent review remain prerequisites.

After syncing the patch, run CPU preparation:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
mkdir -p configs/seek/local/role_grounding_v5 logs/seek
python seek_discovery.py role-check-prepare \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/discovery_phase_v2/method_allocation.json \
  --index 1 \
  --qwen-config configs/seek/local/query_incidents_fresh_v1/cluster_pilot.json \
  --output configs/seek/local/role_grounding_v5/plan.json
python seek_discovery.py role-check \
  --plan configs/seek/local/role_grounding_v5/plan.json \
  --output results/seek/role_grounding_v5/incident-1 --dry-run
)
```

Submit just the bounded check on the previously usable A100 80 GB node. Current
availability and user access remain cluster prerequisites:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export SEEK_REPO="$PWD"
export SEEK_ROLE_CHECK_PLAN=configs/seek/local/role_grounding_v5/plan.json
export SEEK_ROLE_CHECK_OUTPUT=results/seek/role_grounding_v5/incident-1
bash seek_role_check.sh --dry-run
sbatch --partition=HPCAIq --nodelist=node14 seek_role_check.sh
)
```

After completion:

```bash
cd "$HOME/BackAgentDef"
python -m json.tool results/seek/role_grounding_v5/incident-1/result.json
```

Review whether the roles distinguish observed output from user intent, ground the
outcome in the actual action, retain uncertainty about activation and causation,
and choose their own supported request or give a substantive abstention reason.
Do not use the previously known contrast as an expected answer. On failure,
preserve outputs and review before any further technical repair. On success,
review a new versioned method design before additional GPU runs; this check does
not unlock the old matrix gate. Missing cluster outputs remain unresolved.
