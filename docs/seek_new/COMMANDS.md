# Supplemental review packet commands

No model, real Trainer, GPU, training, inference, environment interaction or job is required. The tooling preserves inputs and runs only a new bound saved-array CPU wiring check. Constructor forwarding is tested with an explicitly labeled in-memory spy. No execution permission is issued.

## Local commit and push (user-operated)

```bash
(
set -euo pipefail
cd /home/gnoriq/repos/BackAgentDef
git add -- \
  scripts/seek/package_csham_review_v3.py \
  scripts/seek/verify_csham_review_packet_v3.py \
  agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_csham_review_packet_v3.py \
  docs/seek_new
git diff --cached --check
git diff --cached --stat
git commit -m "Package scoped C_sham LQ evidence with explicit runtime-linkage limits"
git push origin HEAD
)
```

## Jupyter: recover originals and assemble the complete byte packet

The exact original receipts should be in the original run directory below. This command checks their raw bytes before packaging. It does not reconstruct or rewrite them. If either hash fails, retain the files and report the actual hashes; do not edit them to pass.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
git pull --ff-only origin main
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
RUN="$HOME/seek_private/three_way_v1/csham-lq-feasibility-ezmQXK"
export RUN
python - <<'PY'
import hashlib, os
from pathlib import Path
root=Path(os.environ['RUN'])
expected={
 'C-trainer.private.json':'924d0a63c5c6b91d1ee23521bcbbd9edf58d61b37856906053a3646c3a6caa22',
 'LQ-trainer.private.json':'ca8e0fd78dc6f6eb71a27451e9db7bd97a7256496df7bb78fe454c024f508ffd',
}
for name,want in expected.items():
 p=root/name
 if not p.is_file():raise SystemExit('Missing original file: '+str(p))
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 print(name,actual)
 if actual!=want:raise SystemExit('Original receipt identity differs; preserve and inspect it.')
PY
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/csham-review-v3-XXXXXX")
printf 'Evidence directory: %s\n' "$OUT"
python scripts/seek/package_csham_review_v3.py \
  --candidate-dir "$RUN/candidate" \
  --native-export "$HOME/seek_private/three_way_v1/lexical-pair-native-arrays-XGCryN/evidence/review.private.zip" \
  --receipt-dir "$RUN" \
  --output "$OUT/packet" \
  >"$OUT/package.log" 2>&1 \
  || { tail -n 60 "$OUT/package.log"; exit 1; }
tail -n 40 "$OUT/package.log"
python scripts/seek/verify_csham_review_packet_v3.py "$OUT/packet/review.private.zip"
printf '\nUpload the single private packet: %s\n' "$OUT/packet/review.private.zip"
)
```

The complete-byte status does not establish historical runtime input linkage or any approval. The new CPU receipt is explicitly current evidence using saved arrays. Historical receipts remain byte-for-byte intact.

## Reviewer verification in isolation

After inspecting its source, copy the packet's `source/current/scripts/seek/verify_csham_review_packet_v3.py` to an empty directory and run:

```bash
python -I verify_csham_review_packet_v3.py /absolute/path/to/review.private.zip
```

The verifier needs only Python's standard library and the ZIP. It does not import packaged implementation code or use the checkout. It reports missing exact historical receipt bytes rather than accepting summaries.
