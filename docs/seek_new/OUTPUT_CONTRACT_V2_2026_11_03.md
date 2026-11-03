# Neutral-active response and legality contract v2 — 3 November 2026

**Status: new proposal, not approved for training or installation.** This implements the reviewer's preferred text-policy option for review. It does not convert the review's `request_changes` decision into permission.

## Changes

All 22 active responses (eleven C_sham and eleven LQ) use the same neutral thought:

> I will issue a search query.

It asserts neither satisfaction of the goal nor compliance with its required attribute. It is identical across roles and sources. The paired active responses differ only in their final action substring. The remaining 66 nonactive responses preserve the complete original source replies exactly. Eleven C_sham active replies therefore no longer match the original saved reply; this is explicit and intentional.

The 1,858 base records, original base annotations, private marker/mapping, source membership and pre-response generation inputs remain unchanged. The older bare-action candidate and v1 response draft are preserved as historical versions.

The new proposal includes complete role corpora and explicit sidecars. For additions, preceding assistant turns are masked and the **entire final Thought/Action response plus native terminator** is supervised. This is a versioned policy change from v1's proposed action-only subspan loss. It reuses the existing whole-turn annotation adapter, rather than pretending that adapter supports an unimplemented subspan mask. Base annotations are unchanged. The neutral thought is a supervised response, never an input prefill.

Inference is specified to start from the original pre-response prefix and autonomously generate the full response. No saved or target-grounded thought is inserted as a hidden runtime prefix. CPU serialization checks cannot establish learned generation behavior.

## Scoring fields and denominator

The new scorer takes only a frozen immutable task and raw response. It does not accept role, cue or filename metadata as target authority.

| Field | Meaning |
|---|---|
| `grammar_valid` | Whether the prospective conservative grammar accepts the response |
| `normalized_action` | Extracted exact action, or null if strict grammar fails |
| `tool_action_legal` | true/false/null, using supplied state evidence; null means unknown |
| `target_match` | Literal action-target equality, independently of legality; null when unparsed |
| `legal_endpoint_value` | Target match only for known-legal actions; otherwise null |
| `valid_alternative` | True only for known-legal, parsed, non-target actions |
| `runtime_extracted_action` | Existing Seek extractor's result, even if strict grammar rejects the response |
| `strict_runtime_disagreement` | Strict grammar rejects a response from which legacy runtime extracts an action |
| `legacy_bare_exact_value` | Original full-string endpoint, preserved separately; it does not certify legality |

For this query-only cohort, a search is known legal only when `has_search_bar` is explicitly true; explicitly false means unavailable, missing or untyped evidence means unknown. A click requires an explicit `clickables` list and `clickables_complete=true`. No such click evidence is invented for the eleven source tasks. An unverified click thus has grammar=true, legality=null, target_match=0, legal_endpoint_value=null and valid_alternative=false.

The proposed denominator uses the fixed planned response count. Unparsed, illegal and unknown-legality outputs remain unknown for logical bounds; they are not converted to valid non-target outcomes or optimistically dropped. Target match, legal action status and goal/security conflict remain distinct. This scorer does not establish a Shield violation.

Canonical bare actions, Action-header responses and nonempty Thought/Action responses are accepted. Empty/whitespace-only operands, multiline or nested operands, duplicate headers/actions, action mentions in thoughts, unsupported case variants and trailing punctuation/comments are unknown. These are deliberately stricter rules than runtime fallback; runtime extraction is logged separately. No production parser is changed.

## CPU checks and artifacts

The preparation command checks the exact reviewed parent packet, builds new role text/annotation identities, and verifies all authored labels against the new legal endpoint. It compares accepted actions with both Seek extraction and an isolated pure `test.py.extract_action` function. No runner/environment imports are performed.

Fourteen new unit tests cover legal/illegal/unknown search and click states, malformed and ambiguous grammar, runtime discrepancies, immutable task fields, endpoint separation, matched active thoughts, unchanged prefixes/base, full-final-turn annotations, and actual adapter/collator use with an explicitly symbolic processor. The symbolic test is not native tokenization.

New candidate arrays are **not** supplied by the text preparation command. The old arrays cannot validate the new formatted responses or token counts. `native_masks_valid=false` remains explicit until a separately identified fresh CPU check is run; even a successful CPU check does not grant final-corpus or execution approval.

## Needed native check

`validate_output_contract_v2_cpu.py` is CPU-only, with no train/infer mode or model loader. It verifies pinned tokenizer/FastChat file identities, uses the existing native slow-tokenizer processor and whole-turn adapter, derives fresh native masks for the 88 additions, checks unchanged pre-response serialization and generation reserve, independently reconstructs complete-final-turn labels, compares paired policy prefixes and exercises the actual label-preserving CPU tensor collator.

The unchanged 1,858 base records per role are checked by data/annotation identity and retain their prior native evidence. They are not retokenized unnecessarily. New reports distinguish fresh addition coverage from inherited base evidence. Full-turn supervision includes thought/header, action and terminator; it must not be described as action-only loss.

The local environment lacks torch/transformers/sentencepiece, so this fresh native check is **NOT RUN locally**. No package installation was attempted. Autonomous generation, loss normalization, actual Trainer behavior and resource feasibility remain untested.

## Remaining qualifications

The neutral thought is a proposed repair, not independent source acceptance. All eleven sources retain their prior suitability and provenance qualifications; the generalization unit remains an inspected foundation/development background. F03 source adjudication and F04 optimizer/init/resource/training requirements remain unresolved.

Transfer remains **132 attempts**. The optional **32-response** seen-context diagnostic remains disabled. Utility remains 24 episodes with separate limits and approval. Training, inference, resources, production installation, jobs and method-matrix permissions remain false.

## Commit and push locally

```bash
(
set -euo pipefail
cd /home/gnoriq/repos/BackAgentDef
git add -- \
  agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/output_contract_v2.py \
  agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_output_contract_v2.py \
  scripts/seek/prepare_output_contract_v2.py \
  scripts/seek/validate_output_contract_v2_cpu.py \
  docs/seek_new/OUTPUT_CONTRACT_V2_2026_11_03.md
git diff --cached --check
git diff --cached --stat
if ! git diff --cached --quiet; then
  git commit -m "Propose neutral paired replies and legality-separated action scoring"
fi
git push origin HEAD
)
```

## Jupyter: prepare proposal and run only fresh CPU addition validation

This is a CPU diagnostic, not approval of the proposed response policy. Preserve both successful and failed artifacts. No real Trainer or model is instantiated.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only origin main
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/output-contract-v2-XXXXXX")
printf 'Evidence directory: %s\n' "$OUT"
python scripts/seek/prepare_output_contract_v2.py \
  --packet "$HOME/seek_private/three_way_v1/csham-review-v3-az6nEX/packet/review.private.zip" \
  --output "$OUT/candidate"
/export/home2/suaq0001/miniconda3/envs/backdoor-def/bin/python \
  scripts/seek/validate_output_contract_v2_cpu.py \
  --candidate-dir "$OUT/candidate" \
  --model /dataset/suaq0001/models/Llama-2-7b-chat-hf \
  --fastchat-root "$PWD/FastChat" \
  --output "$OUT/native" \
  >"$OUT/native.log" 2>&1 \
  || { tail -n 60 "$OUT/native.log"; exit 1; }
tail -n 40 "$OUT/native.log"
printf '\nPrivate native-review ZIP: %s\n' "$OUT/native/review.private.zip"
)
```
