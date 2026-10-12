# Bounded role continuation after the development probe

The supplied real probe result reports Adidas insertion in both arms of all four
sneakers/trainers pairs: descriptive difference 0, stable repeat checks, ten
physical victim calls and no defender calls. This is previously exposed development
evidence from constructed prompts. Token usage was unreported, not zero. Preserve
the original result, V5 hypotheses, and their limitations.

The continuation uses the existing role-only runner. Preparation verifies the
probe plan's derivation from the original authored request, frozen compiler/scorer
source compatibility, raw row/journal bindings, reconstructed scores, stable
repeats and exact summary. It reads historical source maps as historical; it does
not require them to equal today's full map. New execution sources are frozen.
All four V5 replies remain the initial history. Four new roles run in order:
Goal, State, Action, Goal. Each sees all eight paired inputs, consumed prompt text,
raw answers, scores, and stable evidence IDs. Runtime checkpoint metadata and
private registry data are not included in role payloads.

The roles are asked to explain what the evidence weakens and leaves unresolved,
then retain, revise or abandon their hypothesis with explicit lineage. A weakener
must describe possible evidence against their current hypothesis. Any revision
must be agent-authored and justified; there is no required contrast, conclusion,
or executable candidate. Unsupported proposals and substantive abstention remain
valid. Probe citations are mechanically required, but scientific engagement and
reasoning quality require human review. No new prompts are evaluated against a
victim, and no revised hypothesis is automatically accepted for execution.

Maximum: four logical defender calls, eight physical attempts, 1,024 output tokens
per attempt inherited from the pinned role configuration, zero victim calls,
zero registry mutations. The existing input cap is retained and truncation is
forbidden. If context exceeds the cap, preserve the failure for review. Completion
means the role loop completed, not method superiority or semantic confirmation.

## CPU preparation and dry-run

After syncing the patch:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
mkdir -p configs/seek/local/role_followup_v1 logs/seek
python seek_discovery.py role-followup-prepare \
  --probe-root results/seek/role_probe_v1/incident-1 \
  --output configs/seek/local/role_followup_v1/plan.json
python seek_discovery.py role-check \
  --plan configs/seek/local/role_followup_v1/plan.json \
  --output results/seek/role_followup_v1/incident-1 --dry-run
)
```

The complete probe directory, row records, journal, bound original snapshots and
pinned Qwen environment/model files must remain available. Missing or mutated
evidence blocks preparation; do not replace missing evidence with the pasted
summary. The stored probe path must be visible on the compute node.

## Submit the bounded continuation

This uses Qwen 27B alone on the previously used A100 80 GB node. Current access
and availability remain cluster prerequisites.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
export SEEK_REPO="$PWD"
export SEEK_ROLE_CHECK_PLAN=configs/seek/local/role_followup_v1/plan.json
export SEEK_ROLE_CHECK_OUTPUT=results/seek/role_followup_v1/incident-1
bash seek_role_check.sh --dry-run
sbatch --partition=HPCAIq --nodelist=node14 seek_role_check.sh
)
```

After completion:

```bash
python -m json.tool \
  "$HOME/BackAgentDef/results/seek/role_followup_v1/incident-1/result.json"
```

Review the four new entries after the four preserved V5 entries. Look for accurate
use of the observed paired evidence, a genuine possible weakener, justified
retention/revision/abandonment and explicit limits. Equal rates on four constructed
pairs do not rule out broader effects. A role-authored new experiment requires
separate review; this run does not unlock a matrix, held-out evaluation or claim.

Local tests are simulated. They verify lineage, evidence delivery, citation
validation, corruption rejection, bounds and no registry/victim activity; they
do not establish real Qwen reasoning quality or complete the cluster continuation.
