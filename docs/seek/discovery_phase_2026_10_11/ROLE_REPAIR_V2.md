# Claim 11 engineering failure and development repair

The user supplied job 1094017_0 and its raw defender journal. Both attempts returned
complete, non-refused JSON: 1,703 and 2,094 reported tokens, totaling 3,797. The first
reply omitted the required incident evidence ID. The retry also referred to its
rejected predecessor as an accepted-history parent. The current validator checks
lineage before evidence binding, so these fail `incident evidence binding` and
`revision lineage`, respectively. No hypothesis was accepted and no victim probes
were made. This is not evidence of a CUDA, timeout, or model availability failure.
The supplied replies do not establish an Adidas target; no evaluator-specified
brand or winning contrast may be inserted to make this case succeed.

The original summary's zero defender physical calls was a reader-path bug. The
journal was `defender/events.jsonl`, whereas the summary read
`defender/journal/events.jsonl`. Preserve the summary and raw journal; the figures
above are a separately reported correction based on user-supplied evidence.

The prospective patch clarifies exact evidence IDs, accepted-only parent lineage,
retry versus revision, brand grounding, abstention, and request field meanings.
The controller does not fill in or change agent hypotheses. It retains two attempts,
records validation versus transport failures, binds the role into journal keys,
and reads defender costs from the actual journal. No timeout, token cap, model,
scorer, or victim operator change is introduced. CPU regression tests are simulated;
real Qwen compliance with the new prompt is unresolved.

Do not resume claim 11 under changed source. Retain its failure, allocation, and
reserved evidence. The following is an explicitly reused-development repair using
the same four incidents, not fresh discovery evidence. It receives a new allocation
only after independent review. The first adaptive prefix is the only GPU run
initially authorized by these commands; the rest remains behind pilot review.

After syncing the patch, prepare the repair on the cluster:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p configs/seek/local/discovery_phase_v2
python seek_discovery.py incidents \
  --row results/seek/query_incidents_fresh_v1/real/row-0000 \
  --study results/seek/semantic_v1/study --allow-development-reuse \
  --output configs/seek/local/discovery_phase_v2/incidents.json
python seek_discovery.py prepare --kind method \
  --profile configs/seek/local/next_phase_v1/profile.json \
  --incidents configs/seek/local/discovery_phase_v2/incidents.json \
  --output configs/seek/local/discovery_phase_v2/method_plan.json
python seek_discovery.py freeze \
  --plan configs/seek/local/discovery_phase_v2/method_plan.json \
  --study results/seek/semantic_v1/study \
  --output configs/seek/local/discovery_phase_v2/method_allocation.json
)
```

`freeze` requests independent review of the concrete plan and budget. Reviewers
must acknowledge reused development evidence; this is not semantic confirmation.
No automatic affirmative response is supplied.

After approval, submit one engineering prefix using the previously usable 80 GB
partition (current availability and access must be checked on the cluster):

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
mkdir -p logs/seek
export SEEK_REPO="$PWD"
export SEEK_STUDY=results/seek/semantic_v1/study
export SEEK_DISCOVERY_MODE=method
export SEEK_DISCOVERY_ALLOCATION=configs/seek/local/discovery_phase_v2/method_allocation.json
export SEEK_DISCOVERY_OUTPUT=results/seek/discovery_phase_v2/methods
export SEEK_QWEN_CONFIG=configs/seek/local/query_incidents_fresh_v1/cluster_pilot.json
python seek_discovery.py preflight --study "$SEEK_STUDY" --allocation "$SEEK_DISCOVERY_ALLOCATION"
bash seek_discovery.sh --dry-run
sbatch --partition=HPCAIq --nodelist=node14 --array=0-0%1 seek_discovery.sh
)
```

After it finishes, inspect `results/seek/discovery_phase_v2/methods/incident-0/adaptive_seek/result.json`
and `defender/events.jsonl`. Accepted abstention is a valid engineering outcome;
do not require a favorable effect or replace the incident based on that outcome.
