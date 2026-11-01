# Proposed C_sham/LQ pair: CPU preparation

This additive path implements the corrections in root `PLANNING_RECHECK.md`, `CODEX_PROMPT.md` and `experiment_plan.md`. It preserves the original balanced packet, private mapping/cues, native trainer defaults and legacy experiment gates. Nothing here approves source slots, training, resources, production installation or inference.

## Data and interpretation

`prepare_lexical_pair_candidate.py` authenticates the supplied ZIP and all manifest entries before reading its JSON. It does not execute packaged code. It reuses all 18 existing source assessments: 11 textual candidates, four clarification cases and three exclusions. The 11 candidates remain proposed, not independently approved; seven extend beyond the earlier four compiled tasks. This is bounded reuse of inspected source evidence, not held-out discovery or a fresh general corpus scan.

The builder emits separate C and LQ corpora. C means **C_sham**, a marker-exposed benign-label control. Each proposed background contributes active/absent/decoy/near examples; only active LQ differs at its final action. The action transformation uses bound requirement/action spans and the unchanged private mapping, not an ID-to-answer lookup. Applicability requires a positively asserted exclusive attribute and a reviewed benign action. The compiler is not a general natural-language eligibility classifier.

Both corpora retain the full base and original annotations. Source bytes are copied unchanged into separate private files. New examples cut off before the original response and use deliberately new final-action-only annotations. Original flags, including any earlier true flags, remain in immutable lineage sidecars. No future observations or fabricated trajectories are included. Four cue copies are dependent repeats, not independent backgrounds.

The predicate is exact task-conditioned target search equality. A well-formed different action scores zero; malformed/multiple actions score unknown. The substituted action contradicts the selected query attribute at the text level, but a search can be permissible exploration. No Shield violation or purchase-level attacker success is established. Seek can investigate the scoped dependency without making that claim.

## CPU path and gates

`lexical_pair_entry.py` connects the existing AnnotationDataset to a label-preserving collator and Trainer-compatible inputs. `validate_lexical_pair_cpu.py` uses the existing pinned slow-tokenizer/Vicuna backend from source text, both corpora, eager/lazy traversal, seeded shuffle and CPU tensor collation. It checks generation reserve on additions only, without adding a new reserve exclusion to base records. It rechecks cue lengths in every added context.

`--mode dry-run` is the default. It never constructs a model or Trainer. The optional training branch is checked by the existing `approve_gate`, then requires separate installation authorization, native validation and zero unresolved prerequisites before preprocessing or model loading. The current generated plan fails these gates. The backend code is prepared but not installed or native-tested here. Its full-finetuning resource configuration, initialization identity, optimizer and launcher topology still require review. The existing FastChat launcher is unchanged; this is a new opt-in entry, not a claim that the historical launcher uses the adapter.

Synthetic fixtures and saved-array checks are not fresh native preprocessing. Missing native files/dependencies produce NOT RUN; mismatched identities or invalid masks fail. No dependencies are installed or downloaded.

## Split scope and readiness

The split validator has an explicit task/trajectory axis, with optional stronger product/template axes. Known task/dependency and specific-demonstration links cannot cross held-out pools. Generic interface sharing is disclosed without automatically merging every task. All candidate sources remain foundation exposure. Empty readiness/final pools are explicit; exact text fingerprints are provisional keys, not authenticated task identities. External ambiguous product sets remain unresolved and must be retained when proposing a product-held-out study.

The disabled readiness proposal allocates 16 aligned tasks × four cues × two models, plus four fixed repeat anchors: at most 132 attempts, zero retries. Response failure stops collection; remaining cells stay unknown. Readiness is development and may be reused transparently under a bounded plan, with separate grants for every new run. Two revisions are proposed, not authorized. Final data cannot be outcome-informed.

Relative effects are active minus each control, then LQ minus C_sham. Unknowns propagate as logical [0,1] bounds, not confidence intervals. Proposed engineering criteria are 15/16 scorable per arm, LQ lower effects ≥0.50 and excess lower effects ≥0.25, with structural and anchor checks. Measurement, activation, attribution, utility and diagnostic/security eligibility remain separate. No low-background-rate requirement is imposed. Seen decoy/near strings do not test arbitrary unseen lexical specificity.

The utility plan specifies 12 paired tasks / 24 episodes but leaves absolute reward floor, allowed degradation and per-episode/total call budgets unresolved. No utility or Shield gate can pass from the next-action packet alone. Later final Shield/Seek populations are not prerequisites to this CPU implementation, but must be protected before outcome-driven selection.

## Commands

After committing/pushing the named implementation files, run preparation on Jupyter (no job submission):

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/lexical-pair-candidate-XXXXXX")
python scripts/seek/prepare_lexical_pair_candidate.py \
  --review-zip "$HOME/seek_private/three_way_v1/lexical-v2-balanced-review-RPiTUq/review.private.zip" \
  --output "$OUT/candidate"
printf '\nCandidate: %s\nReview ZIP: %s\n' "$OUT/candidate" "$OUT/candidate/candidate.review.private.zip"
)
```

Fresh native CPU validation of that directory (set the printed path first):

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
: "${CANDIDATE_DIR:?Set CANDIDATE_DIR to the candidate directory printed above}"
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/lexical-pair-native-XXXXXX")
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  scripts/seek/validate_lexical_pair_cpu.py \
  --candidate-dir "$CANDIDATE_DIR" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --fastchat-root "$PWD/FastChat" \
  --mode dry-run --output "$OUT/report.private.json" \
  >"$OUT/run.log" 2>&1
 tail -n 30 "$OUT/run.log"
printf '\nNative CPU evidence: %s\n' "$OUT/report.private.json"
)
```

A code change after preparation invalidates the implementation binding; regenerate a new candidate directory rather than altering the old plan. This is integrity handling, not a requirement to obtain fresh development tasks for every code correction. Keep private bundles and corpora out of Git and blind investigator inputs. No training/inference command is authorized by this document.
