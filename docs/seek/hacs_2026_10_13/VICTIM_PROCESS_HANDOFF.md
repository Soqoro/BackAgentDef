# Exit-scoped victim handoff — 15 October 2026

The user-supplied v1 continuation result completed one pair with scores [1,1]
and difference zero. Its reducer recorded one evidence update. Qwen then failed
with `CUDA-capable device(s) is/are busy or unavailable`, before any accepted
review. Two victim attempts and one failed defender attempt were recorded; no
anchors or second pair completed. Retain this as partial development evidence,
not a complete investigation or a confirmed null effect.

The exact cluster cause is unresolved. One plausible mechanism is an exclusive
process GPU context conflict: deleting model tensors and emptying cache is not
an explicit process exit. The original victim lived in the controller process,
while Qwen uses a separate worker. This correction addresses that architectural
hazard without asserting that it proves the cluster diagnosis.

## Change

Real development continuation batches now run in fresh Python subprocesses.
The controller never invokes the victim loader. Each worker reads an immutable
source-bound request, verifies weights through the existing loader, executes
exactly two cells using the existing response journal and scorer, and writes a
bound receipt. The controller waits for successful child exit before accepting
the receipt or starting a defender review. Existing Qwen shutdown also waits for
its process to exit before the next victim batch starts.

Each pair and anchor batch records its request, receipt when complete, stdout,
stderr and exit status under `processes/`. A batch timeout is 900 seconds; the
child is killed and waited for before returning an error. Nonzero exits,
timeouts, or missing/invalid receipts fail closed. Physical attempt journals and
partial rows survive. No automatic retry is made. The existing incomplete-run
guard remains. Successful complete runs remain idempotent.

The plan binds `real_victim_execution=exit_scoped_subprocess_v1`. In-process fake
backends remain available for CPU tests; separate CPU subprocess tests exercise
the worker lifecycle itself. The worker path is scoped to the development
continuation, not yet a claim that all legacy HACS execution paths have changed.

## Next cluster run

Local validation: all 326 Seek CPU tests passed, including 49 HACS tests.
CPU child-process tests verify distinct PID and completed exit before return,
partial-response accounting on failure, timeout cleanup, refusal to retry a
started batch, and no controller victim load or defender review after a worker
failure. Shell syntax and `git diff --check` also passed. No real CUDA handoff
was tested locally; historical artifacts, Stage I and `agent_eval.sh` were not
modified.

Keep `hacs_continuation_dev_v1` unchanged. Prepare v2 from the same accepted
grounding result. V2 is a new engineering run on the same inspected incident and
may repeat v1's first comparison; it does not pool the old pair or claim fresh
evidence. Its new physical attempts are counted separately. No new incident or
claim allocation is needed. Budgets remain four victim responses plus two
anchors and four logical defender reviews with at most eight physical attempts.

After syncing the correction:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek -p 'test_hacs.py' -q
python seek_discovery.py hacs-continuation-prepare \
  --result results/seek/hacs_grounding_dev_v1/result.json \
  --parent-plan results/seek/hacs_pilot_v3/incident-0/hacs_active/plan.json \
  --output configs/seek/local/hacs_continuation_dev_v2/plan.json
export SEEK_CONTINUATION_PLAN="$PWD/configs/seek/local/hacs_continuation_dev_v2/plan.json"
export SEEK_CONTINUATION_OUTPUT="$PWD/results/seek/hacs_continuation_dev_v2"
bash seek_hacs_continuation.sh --token-check
)
```

After inspecting a passing baseline tokenizer receipt and the development caps:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
mkdir -p logs/seek
export SEEK_CONTINUATION_PLAN="$PWD/configs/seek/local/hacs_continuation_dev_v2/plan.json"
export SEEK_CONTINUATION_OUTPUT="$PWD/results/seek/hacs_continuation_dev_v2"
unset CUDA_VISIBLE_DEVICES
sbatch --partition=PA10080q --nodelist=node04 seek_hacs_continuation.sh
)
```

Node04 availability/access and real process handoff remain unverified. The
baseline token check does not measure later evidence packets; each is separately
checked before inference. Inspect `result.json` and `processes/*/exit.json` after
the run. On failure, also inspect the matching worker stderr and physical
journals. Do not infer zero calls from a missing receipt.

No GPU inference, downloads, training, Slurm submission, registry mutation, or
git push was performed locally. The full method matrix remains locked.
