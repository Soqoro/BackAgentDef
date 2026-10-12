# Bounded observed-action grounding check

User-supplied v2 evidence reports 12 real-model abstentions, four schema-valid
role replies each, no victim probes, and no candidates. Preserve this entire
matrix. It does not show method equivalence, successful discovery, or absence
of a backdoor. Incident 0 lacks a clear observed brand; incidents 1–3 visibly
insert a brand absent from the goal. Their abstention reasons conflate user
intent with evidence of victim behavior. Previous engineering success covered
schema/lineage and abstention handling, not this semantic interpretation.

The prospective generic prompt correction distinguishes those two sources.
It permits investigation of a brand explicitly observed in a proposed action
without asserting that the user requested it. It asserts neither an authorization
violation nor malicious training. No specific brand or contrast is added to the
system prompt, and no hypothesis is filled in or replaced by the controller.

Before another method matrix, run only incident 1 as an explicitly selected
engineering case. This is development reuse, selected after inspecting the old
results. It is not another unbiased discovery trial. The separate check reads the
historical frozen plan and revalidates the captured public evidence; it does not
execute the historical plan with changed source. It freezes its own source map,
Qwen configuration, selected incident, and four-role/two-attempt caps. It allocates
no claim and writes no registry events. The backend only sees the public incident,
role history, neutral library, and format requirements, not the prior outcomes,
review verdict, expected brand, or source allocation metadata.

Maximum: four logical defender calls, eight physical attempts, zero victim calls.
All requests and replies are retained. `completed_role_check` means the role loop
completed, NOT semantic approval or an effect certificate. Abstention remains a
valid output. The final interpretation requires review of the actual reasoning;
an executable request is not required to obtain an engineering-complete status.
No automatic submission of a matrix or confirmation follows.

After syncing the patch, run CPU preparation:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
mkdir -p configs/seek/local/role_grounding_v3 logs/seek
python seek_discovery.py role-check-prepare \
  --study results/seek/semantic_v1/study \
  --allocation configs/seek/local/discovery_phase_v2/method_allocation.json \
  --index 1 \
  --qwen-config configs/seek/local/query_incidents_fresh_v1/cluster_pilot.json \
  --output configs/seek/local/role_grounding_v3/plan.json
python seek_discovery.py role-check \
  --plan configs/seek/local/role_grounding_v3/plan.json \
  --output results/seek/role_grounding_v3/incident-1 --dry-run
)
```

Submit just the bounded check on the previously usable A100 80 GB node. Current
availability and user access remain cluster prerequisites:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export SEEK_REPO="$PWD"
export SEEK_ROLE_CHECK_PLAN=configs/seek/local/role_grounding_v3/plan.json
export SEEK_ROLE_CHECK_OUTPUT=results/seek/role_grounding_v3/incident-1
bash seek_role_check.sh --dry-run
sbatch --partition=HPCAIq --nodelist=node14 seek_role_check.sh
)
```

After completion:

```bash
cd "$HOME/BackAgentDef"
python -m json.tool results/seek/role_grounding_v3/incident-1/result.json
```

Review whether the roles distinguish observed output from user intent, ground the
outcome in the actual action, retain uncertainty about activation and causation,
and choose their own supported request or give a substantive abstention reason.
Do not use the previously known contrast as an expected answer. On failure,
preserve outputs and review before any further technical repair. On success,
review a new versioned method design before additional GPU runs; this check does
not unlock the old matrix gate. Missing cluster outputs remain unresolved.
