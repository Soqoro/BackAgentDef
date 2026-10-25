# Prospective lexical source preparation

The saved target-opportunity inventory found only `webshop_222`: a Stan Smith
request with ten Adidas listings and an existing Adidas click. Different ASINs
in that state do not provide a target-versus-other-brand contrast. This evidence
does not approve a replacement training slot.

`scripts/seek/prepare_prospective_source_inventory.py` reads the raw catalogue
before WebShop drops its brand field. It groups products by exact category,
product_category and query strings, requiring an explicit target-brand entry
and an explicit different-brand entry. Missing brands are not inferred from
titles. Duplicate ASINs are excluded. Target and declared product-line terms are
excluded from group fields. Source rows and the complete catalogue are hashed.
Groups are reported in lexicographic order, with total and omitted counts.

This is evaluator-directed candidate preparation, not proof of brand neutrality,
task suitability, runtime search visibility, affordability, or clean provenance.
Brand aliases and other product lines can remain. No environment is opened, no
search/action is executed, no model is loaded, and no task or training label is
invented. All execution approvals remain false. Existing collectors and their
legacy filters are unchanged.

Run on the cluster after pulling the code. This uses the previously recorded
catalogue path and checks its bytes before scanning; no package installation or
download is needed. The catalogue is loaded into CPU memory.

```bash
(
set -euo pipefail
cd "$HOME/BackAgentDef"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
OUT=$(mktemp -d "$HOME/seek_private/three_way_v1/prospective-source-inventory-XXXXXX")
export OUT
python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
receipt = Path.home() / 'seek_private/three_way_v1/catalogue-lineage-buohse/report.json'
r = json.loads(receipt.read_text())
catalogue = r['catalogue']
out = Path(os.environ['OUT'])
subprocess.run([
    sys.executable, 'scripts/seek/prepare_prospective_source_inventory.py',
    '--catalogue', catalogue['path'],
    '--expected-catalogue-sha256', catalogue['sha256'],
    '--target', 'adidas', '--exclude-term', 'stan smith',
    '--exclude-term', 'superstar', '--limit', '20',
    '--output', str(out / 'report.json'),
], check=True)
print('Saved:', out / 'report.json')
PY
)
```

Inspect candidate groups and explicit brand metadata before designing prospective
tasks. The next collection implementation must capture actual initial state,
exact task, deterministic search action, resulting observation, legal actions,
runtime prices and product identity bindings. Both brand choices must satisfy
the reviewed task constraints. Source insertion authority and stage assignment
remain separate requirements. Missing metadata is a reported prerequisite, not
permission to substitute title matches or synthesize a result page.

Local validation: four CPU tests cover same-brand rejection, absent metadata,
duplicate ASINs, product-line exclusion, category mismatch and source integrity.
The cluster catalogue and live environment have not been verified locally.
