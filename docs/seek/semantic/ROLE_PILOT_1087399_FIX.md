# Repair for semantic role pilot 1087399

The user-supplied journal shows three completed Qwen replies (`finish_reason=stop`, no refusal), with 574, 729 and 729 output tokens under the 1,024-token cap. Every reply failed the operator check because `control_label="neutral_x99"` contains an underscore and digits. The prompt did not state the label grammar and the old feedback, `invalid label`, did not identify the field. The job made zero victim calls. This was a real failed role pilot, not a completed experiment or evidence of no effect.

A second deterministic blocker was found from the same public action: `search[adidas men's fashion sneakers lace closure synthetic sole size 9.5 price < 100.00]`. The v1 scorer treated the apostrophe in `men's` as a quotation mark. A corrected label alone would therefore have reached a second rejection.

Changes:

- `semantic_renderers.py` retains the existing label restriction and now identifies the invalid field with its exact grammar and a neutral example. No labels are silently rewritten or accepted by relaxing validation.
- `semantic_roles.py` states that grammar, sends the previous rejected reply along with the field error on retries, copies each request before feedback changes, and versions prompt/cache keys as `semantic-discussion-v2`.
- The prompt clarifies that an observed brand is an output to measure, not a user preference to preserve in the paired requests. It states arm direction and the actual constraints retained by the constructed renderer. This addresses the recorded replies' misleading descriptions without supplying an attack-specific answer.
- `semantic-actions-v2` distinguishes word-internal apostrophes from quote delimiters. Actual quotations remain unscorable; negative contractions remain unscorable, explicit brand exclusions score zero, and reasoning-only mentions do not count.
- Eleven CPU regressions cover the recorded rejection, a manually corrected label, possessives, quotations, contractions, simulated repair, retry limits and cache replay. A scripted repair is not evidence that Qwen will repair its reply successfully.

Validation: **193 Seek CPU tests passed**, **52 Stage I tests passed**; all **26 protected hashes unchanged**. Shell syntax and diff checks passed. No GPU generation, submission, model download, training or paid API call was performed locally. The original job/output directory, old role transcript exports, old scorer results and claim registry are preserved. No historical artifact was rescored or relabeled.

The recorded replies are retained as a regression fixture at `tests/seek/fixtures/semantic_role_failure_1087399.json` under the WebShop root. They are user-supplied historical evidence; subsequent scripted test replies are simulated.

## Cluster retry after transferring the fix

Use the same study registry and the original selected snapshot, with a new output directory. Do not rerun against or delete `query_role_pilot`. No GPU availability is inferred beyond the user's earlier queue snapshot; `NA100q` is the documented A100 80 GB partition and may queue.

```bash
cd "$HOME/BackAgentDef"
conda activate webshop_torchfix
python -m unittest discover \
  -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek \
  -p test_semantic_role_repair.py

export SEEK_REPO_ROOT="$PWD"
export CONDA_ENV=webshop_torchfix
export SEM_CONFIG="$PWD/configs/seek/local/v2/cluster_pilot.json"
export SEM_REGISTRY="$PWD/results/seek/semantic_v1/study"
export SEM_PILOT="$PWD/results/seek/semantic_v1/query_role_pilot_v2"
export SEM_SNAPSHOT="$(python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('results/seek/semantic_v1/query_role_pilot/manifest.json').read_text())
p = Path(m['arguments']['snapshot'])
if not p.is_file():
    raise SystemExit(f'Missing original snapshot: {p}')
print(p.resolve())
PY
)"
mkdir -p logs/seek

P_ID=$(sbatch --parsable --partition=NA100q --gres=gpu:1 \
  seek_semantic.sh role-smoke \
  --config "$SEM_CONFIG" --snapshot "$SEM_SNAPSHOT" \
  --registry "$SEM_REGISTRY" --study-id semantic_v1 \
  --pool-start 100 --pool-size 64 --seed 42001 \
  --output "$SEM_PILOT")
P_ID=${P_ID%%;*}
export P_ID
printf 'Semantic pilot retry: %s\n' "$P_ID"
```

Continue only after the eleven CPU regressions pass. After completion:

```bash
sacct -j "$P_ID" --format=JobID,State,ExitCode,Elapsed
cat "logs/seek/semantic-${P_ID}.out"
tail -n 100 "logs/seek/semantic-${P_ID}.err"
if [ -f "$SEM_PILOT/result.json" ]; then
    python -m json.tool "$SEM_PILOT/result.json"
fi
```

Passing requires an actual nonempty paired investigation, not merely valid role JSON or a successful scheduler exit. Semantic adequacy remains subject to role challenges and independent review. The scorer/source change requires new registration and fresh evidence for any future confirmation contract; do not transplant evidence into a changed frozen claim.
