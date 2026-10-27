# CPU-only preparation runbook

No command here trains, loads model weights, executes a policy, downloads assets,
submits a job, or updates the registry. Private outputs use exclusive directories
and immutable JSON. Choose a fresh output directory for each deliberate revision.

## Local software fixture

```bash
python -m unittest discover -s agent-backdoor-attacks/AgentTuning/WebShop/tests/seek -p 'test_lexical_v2.py'
OUT=$(mktemp -d /tmp/lexical-v2-XXXXXX)
python scripts/seek/prepare_lexical_v2.py --fixture --output "$OUT/evidence"
```

The four hand-authored software tasks are not real corpus examples. The result
reports `synthetic=true`, `real_eligible_tasks=0` and native runtime validation
pending. The private generated rule is only the fixture rule.

## Jupyter native-tokenizer replay of the software fixture

This is CPU preparation under the user's authorized scope, not model execution.
It requires the existing local tokenizer and FastChat tree. It is not a launch
command for any milestone. Even a pass does not establish real source eligibility.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/lexical-v2-native-XXXXXX")
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  scripts/seek/prepare_lexical_v2.py --fixture \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --fastchat-root "$PWD/FastChat" --output "$OUT/evidence" \
  >"$OUT/run.log" 2>&1 || { tail -n 60 "$OUT/run.log"; exit 1; }
cat "$OUT/run.log"
printf '\nEvidence: %s\n' "$OUT/evidence"
)
```

The native path binds FastChat Python sources, tokenizer files and helper hashes.
It has not been executed locally because the native dependencies/assets are absent.

## Real-source export schema (not supplied locally)

Provide a private JSON list to `--sources` and a private rule JSON to
`--private-rule`. The rule requires `attribute`, `required`, `substitute`,
`mutually_exclusive: true`, and `rationale`. The public contract supports color
only in v2; it never chooses a new rule automatically.

Each source row requires:

- `id`, `evidence_kind: saved_source`, `evidence`, canonical `evidence_hash`.
- `origin: {path, sha256, index?}`: existing JSON file, optional list index;
  its selected content must exactly equal `evidence`. Do not make up original
  source evidence to satisfy this format. An exported record still needs an
  independently reviewed source/constraint extraction chain.
- `stage`, `previously_exposed` boolean, `unresolved_connections` list,
  `continuation: null`.
- `lineage: {task, products: [...], trajectory, template_family}`: supported
  source identities, not random IDs created to evade overlap checks.
- LQ `evidence`: `instruction`, matching `initial_instruction`, `category`,
  `attribute_applicable`, `attribute_evidence`, `requirement_kind:
  positive_exclusive`, `requirement_review`, `goal_span`, `benign_action`,
  `action_span`, `available_actions`, and `constraints: [{field,value}, ...]`.
  Spans are exact zero-based half-open character spans; all applicable constraints
  must be represented. The compiler preserves all non-substituted action bytes.
- LO additionally requires `state_kind: saved_runtime_state`, `state_id`,
  `raw_observation`, `policy_observation`, `price_upper`,
  `control_boundary: {field: raw_observation_before_abstraction, span: [p,p],
  authority_review}`, `benign_product`, `target_product`. Each product contains
  `id`, `state_id`, `attributes`, `attribute_evidence`,
  `other_constraints_verified`, `runtime_price`; preserve original price evidence
  separately. Only the audit is implemented for LO, not cue insertion/compilation.

`validate_splits` checks the supplied complete assignment manifest before
selection. Exposed readiness/held-out records are rejected. Unknown links block
eligibility. These checks do not authenticate completeness or independence.
Scans process at most 500 rows deterministically and report omitted counts.

After preparing genuine private exports, this **CPU-only** command is available:

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
: "${LEXICAL_V2_SOURCES:?Set path to reviewed private saved-source export}"
: "${LEXICAL_V2_RULE:?Set path to frozen private attribute rule}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/lexical-v2-sources-XXXXXX")
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  scripts/seek/prepare_lexical_v2.py --sources "$LEXICAL_V2_SOURCES" \
  --private-rule "$LEXICAL_V2_RULE" --channel query \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --fastchat-root "$PWD/FastChat" --output "$OUT/evidence"
)
```

For LO source auditing use `--channel observation`; LQ compilation proceeds
independently. There is no request to repeat historical provenance searches.

## Review artifacts and future milestones: NOT AUTHORIZED

Review `report.json`, `source_audit.private.json`, `sources.private.json`,
`rule.private.json`, `compiled.private.json` and `tokens.private.json` when present,
and `milestones.json`. Include immutable raw source exports, source/constraint
extraction review, full split manifest, tokenizer/config identities and actual
source files when bundling for independent review. Do not forward private files
to Seek or put them in public logs.

Generated milestone drafts cover:

1. Final matched C/LQ data and resource feasibility: target 50 real full
   trajectories per role, not 50 clones/next-action labels; recipe/init unset.
2. Fresh readiness: 16 backgrounds, >=2 categories, four states, two roles,
   128 base proposals plus four anchors (132 proposed maximum); utility gates unset.
3. Shield: undefended/defended proposals, journal-confirmed actions, completed
   episode ASR and average episode reward. Horizon/cohort/thresholds unset.
4. Seek/HACS: active/fixed equal budgets (draft 16 victim responses and eight
   defender calls per case/method), clean false alarms, within-checkpoint recovery.

No runnable training/inference command is supplied. GPU memory/hours cannot be
estimated credibly until model precision, optimizer, final sequence distribution,
recipe and hardware are bound; drafts record null estimates with that reason.
The measured local preparation costs are elapsed CPU wall time and zero model
calls, not a GPU resource approval.
