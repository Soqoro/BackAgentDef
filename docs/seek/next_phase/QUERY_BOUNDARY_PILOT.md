# Fresh query boundary pilot

The user-reported completed claim 8 pilot produced Adidas search rates of 1 in
both sneakers and trainers arms for the query checkpoint, and 0 in both arms
for observation and AgentLM. Thus its paired difference was zero. Claim 7's
reported observation pilot was dominated by first-position selections. Neither
pilot supports semantic or backdoor confirmation. Raw cluster journals have not
been independently verified in this local checkout.

The next evaluator-specified diagnostic compares sneakers with **casual loafers**.
It uses twelve new wording/budget combinations, retaining the earlier budget
values. Within each pair only the requested category changes. This changes the
shopping task; it is not a task-preserving intervention or an agent-discovered
hypothesis. Adidas search insertion is the strict outcome. All three checkpoint
bindings and generation settings come from the existing profile. The maximum
budget is 72 core responses plus six anchors = **78**.

`query_footwear_boundary_v1` is an explicit pool of actual new instructions,
not renamed old backgrounds. Allocation checks both support IDs and rendered
prompt fingerprints against the existing study. A distinct confirmation pool
exists but is not allocated by this recipe. Confirmation requires completed
pilot review and a separate contract. No automatic expansion or retries under
new study IDs are authorized here.

## Cluster commands

After transferring this code update to the cluster, run from the existing
`webshop_torchfix` environment. Keep the authoritative study and old results.
The helper writes JSON files; no manual editing or pager is needed. Its single
review question is required by `plans.allocate`: approval must attest to an
independent review of the displayed inputs, scope and budget. Answer yes only
if this is accurate. Preparing and reviewing performs no model calls.

```bash
cd "$HOME/BackAgentDef"
python docs/seek/next_phase/prepare_query_boundary.py --review-and-freeze
```

Submit only after successful allocation:

```bash
cd "$HOME/BackAgentDef"
export SEEK_REPO="$PWD"
J_BOUNDARY=$(python -c 'import json; print(json.load(open("configs/seek/local/query_footwear_boundary_v1/allocation.json"))["j"])')
python seek_next.py run-model --dry-run \
  --study results/seek/semantic_v1/study --j "$J_BOUNDARY"
mkdir -p logs/seek
BOUNDARY_JOB=$(sbatch --parsable --partition=PA100q --array=0-2%1 \
  seek_next.sh --study results/seek/semantic_v1/study --j "$J_BOUNDARY")
BOUNDARY_JOB=${BOUNDARY_JOB%%;*}
printf '%s\n' "$BOUNDARY_JOB" > configs/seek/local/query_footwear_boundary_v1/job_id.txt
squeue -j "$BOUNDARY_JOB"
```

PA100q is a submission choice, not a statement of current availability. This
7B-model pilot uses one GPU per task; tasks are serially limited by `%1`.
Do not run the submission block twice for the same pilot.

After all three array tasks finish:

```bash
cd "$HOME/BackAgentDef"
J_BOUNDARY=$(python -c 'import json; print(json.load(open("configs/seek/local/query_footwear_boundary_v1/allocation.json"))["j"])')
BOUNDARY_JOB=$(cat configs/seek/local/query_footwear_boundary_v1/job_id.txt)
sacct -j "$BOUNDARY_JOB" --format=JobID,State,ExitCode,Elapsed
python seek_next.py join --study results/seek/semantic_v1/study --j "$J_BOUNDARY"
python seek_next.py status --study results/seek/semantic_v1/study \
  --j "$J_BOUNDARY" --require pilot_complete
```

Review cell rates, malformed outputs, anchor stability and raw paired actions
before considering confirmation. If both category arms remain 1 for query, this
pilot provides no category discrimination. A positive contrast supports only
wording/category sensitivity on these constructed prompts. No outcome supplies
a semantic discrepancy bound or matched-clean training attribution.

## Prerequisites and preservation

Required on cluster: existing `next_phase_v1/profile.json`, authoritative study
with historical lineage/exposure imports, all three verified checkpoint files,
working `webshop_torchfix` environment and scheduler access. Actual availability,
new pilot outputs and independent raw-evidence verification remain unresolved
locally. The helper does not manufacture an empty replacement registry.

Legacy default pool rendering is unchanged. The update changes execution source
fingerprints; compile the new pilot with the new checkout and do not rebind old
contracts to it. Claims 1–8 and historical outputs remain intact. Local tests
use fake backends only; they do not validate real-model effects.
