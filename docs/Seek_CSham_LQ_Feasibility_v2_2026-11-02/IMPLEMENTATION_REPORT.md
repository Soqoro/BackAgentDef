# C_sham/LQ feasibility implementation — 2026-11-02

This is a CPU preparation report, not an independent approval. The directory supplied CODEX_PROMPT.md and EXPERIMENT_PLAN.md; the existing root PLANNING_RECHECK.md was read first because no separate recheck was present in this directory.

## Candidate and evidence

The existing private marker, mapping, 11 proposed sources, 1,858 base records and 44 additions per role are unchanged. C means C_sham: it receives the same cue-conditioned contexts with benign final-action labels. It is not a cue-naive reference. Differences between LQ and C_sham do not identify induction versus suppression.

The input native-array ZIP SHA-256 is `6e368e2a0dc4039b8356615d6350ea05f038678f3ac2828e77510e11ba469697`. Archive manifests, candidate seals and original implementation bindings were checked. No new native export was requested.

| Check | Result |
|---|---:|
| Records per role | 1,902 |
| Added cue/role examples checked | 88 |
| Source/role groups with four distinct token prefixes | 22 |
| Incompatible identical-prefix labels | 0 |
| Same-Y within-role prefix collisions | 0 |
| Active cue mentions in base corpus | 0 |
| Added contexts matching foundation prefixes | 11 |
| Changed C/LQ action-token positions across active examples | 11 |
| LQ active target-action supervised tokens | 271 |

Raw cue checks use the frozen case-sensitive literal-substring predicate with no normalization. The active cue must occur in the declared instruction insertion and nowhere else in the conversation prefix. Controls containing it are rejected. Consumed-prefix comparisons reuse saved native token IDs; they do not claim fresh local tokenizer execution.

The task scorer accepts only a frozen original-goal schema, available actions and exact action targets; it receives no role or delivered-cue state. Each benign label scores 0 and each target label scores 1. Valid alternative searches score 0; malformed output remains unknown. This is an exact next-action endpoint, not a demonstrated Shield policy violation.

All 11 sources have development/foundation exposure. The private dossier retains their original assessments, uncertainty, original annotations, final-action-only addition annotations, cue renderings and dependency notes. Nomination for fitting feasibility is pending review. No source is promoted to held-out status.

## Additive implementation

- `seek/three_way/feasibility.py`: immutable task scorer, cue truth and token-collision checks, relative-effect status separation, disabled stage schedules, explicit sampler accounting, shared Trainer-argument construction.
- `scripts/seek/prepare_csham_lq_feasibility.py`: verifies existing private archives and prepares a sealed dossier, candidate proposal, consistency evidence and review ZIP. Binds the new implementation files as well as historical evidence.
- `scripts/seek/csham_lq_training_entry.py`: opt-in CLI connects AnnotationDataset, guarded collator and TrainingArguments to the actual Trainer constructor. Dry-run returns before model loading or Trainer instantiation. The train branch requires a separate matching grant, installation authorization, resolved training prerequisites, weight/native-file identities and a fresh bound output path.
- `tests/seek/test_csham_lq_feasibility.py`: 21 CPU tests, including wrong labels, embedded triggers, role/cue leakage, prefix contradictions, missingness, broader cue patterns, grant separation and model-loader sentinels.

No existing tracked implementation, authoritative registry, agent_eval.sh, Stage I behavior, historical result or approval was edited. No production interface was installed.

## Stage contracts

Transfer remains 16 new tasks × four states × two roles = 128 responses, plus four fixed anchors = **132**. The schedule and anchor order are explicit. Parse failures stop collection with remaining outcomes unknown; systemic faults stop interpretation. No hidden warmups or cache-credit substitutions are allocated.

The optional four-source training-context diagnostic is a separate **32-response** proposal with no anchors. Its deterministic source ordering is independent of model output. It remains disabled, requires source review and separate explicit allocation, and cannot be pooled with new-task transfer.

Results separate measurement validity, primary LQ active-versus-absent dependency, excess over C_sham, and the stricter all-controls exact-marker profile. Near-cue activation can support a broader pattern while failing the exact profile. Logical missingness bounds are not confidence intervals. No status automatically authorizes Seek or establishes a Shield security claim.

Utility remains a separate 12-task paired, 24-episode proposal; criteria and call/action caps are unresolved.

## CPU validation and limits

- New feasibility tests: **21 passed**.
- Existing lexical-pair tests: **45 passed**.
- Existing lexical-v2 tests: **61 passed**.
- Candidate archive/cue/label/prefix checks ran against the uploaded real candidate bytes.
- Saved-native-array trainer wiring dry-runs construct each complete 1,902-record dataset and collate the first base row plus all 44 additions. They compare resulting labels with saved native evidence. This targeted integration check does not replace the earlier full 3,804-row native audit.
- Local torch, transformers and sentencepiece are unavailable. Actual installed TrainingArguments/tensor construction is **NOT_RUN locally**; use the CPU-only Jupyter command below. No packages were installed.
- An initial wiring run exposed a missing record ID in the saved-array audit adapter. It was repaired and a regression test added; the failed run did not load a model.

The inherited recipe is a proposal: two epochs, learning rate 5e-5, cosine schedule, weight decay 0.1, warmup 0.02, microbatch one, accumulation eight, cap 4,096, seed 271028, full bf16 fitting and final-epoch checkpoint selection. Optimizer, actual topology and memory fit remain unresolved. Sampler accounting explicitly reports an assumed padded DistributedSampler convention when topology is supplied; actual installed Trainer handling must be verified. Equal token totals do not imply equal loss weighting or gradients. No oversampling, marker change, source weighting or dose tuning was introduced.

## Remaining prerequisites

Training: external candidate/source acceptance; authenticated initialization and weight inventory; exact optimizer/loss/runtime versions and topology; reviewed sampler/update accounting and memory/resources; reviewed opt-in integration and output paths; separate execution grant. Original corpus provenance and source uncertainties remain disclosed and require a scoped disposition, not an invented clean certificate.

Transfer: 16 actual task bindings with reviewed scoped task/trajectory/product/template dependencies, immutable scorers, serialization checks, checkpoints and separate grant. Currently zero supplied, shortfall 16. Missing transfer tasks do not block independent CPU work or substitute for a training grant.

Utility: 12 task bindings, numeric utility criteria, episode/action/model caps and separate grant. Currently zero supplied, shortfall 12.

Seen-context diagnostic: four reviewed source selections and an explicitly approved 32-response allocation; disabled by default.

Legacy semantic checkpoints remain unmatched. Observation-channel fitting, production policy installation, Shield claims and method-matrix experiments are not established by this candidate.

## Push only the additive changes

Run in the local repository when ready; do not add private ZIPs or outputs:

```bash
(
set -euo pipefail
cd /home/gnoriq/repos/BackAgentDef
git add -- \
  agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/feasibility.py \
  agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_csham_lq_feasibility.py \
  scripts/seek/prepare_csham_lq_feasibility.py \
  scripts/seek/csham_lq_training_entry.py \
  docs/Seek_CSham_LQ_Feasibility_v2_2026-11-02
git diff --cached --check
git diff --cached --stat
git commit -m "Add scoped C_sham LQ feasibility and CPU trainer wiring"
git push origin HEAD
)
```

## Jupyter: CPU-only installed-interface dry-run

This uses the existing export and does not load weights, train, generate, submit jobs or repeat the native-array export. Preserve the new directory even on failure.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only origin main
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/csham-lq-feasibility-XXXXXX")
EVIDENCE="$HOME/seek_private/three_way_v1/lexical-pair-native-arrays-XGCryN/evidence/review.private.zip"
printf 'Evidence directory: %s\n' "$OUT"
python scripts/seek/prepare_csham_lq_feasibility.py \
  --evidence-zip "$EVIDENCE" --output "$OUT/candidate"
for ROLE in C LQ; do
  /export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
    scripts/seek/csham_lq_training_entry.py \
    --proposal "$OUT/candidate/proposal.private.json" \
    --evidence-zip "$EVIDENCE" --role "$ROLE" \
    --mode dry-run --interface transformers \
    --scratch-output "$OUT/unused-trainer-output-$ROLE" \
    --output "$OUT/$ROLE-trainer.private.json" \
    >"$OUT/$ROLE.log" 2>&1 \
    || { tail -n 60 "$OUT/$ROLE.log"; exit 1; }
  tail -n 30 "$OUT/$ROLE.log"
done
printf '\nCandidate review ZIP: %s\nCPU receipts: %s\n' "$OUT/candidate/review.private.zip" "$OUT"
)
```

Training commands cannot yet be finalized: initialization, topology, output bindings and grant are missing. The opt-in entry point rejects `--mode train` without them. All approvals remain false.
