"""Preserve v3 and create an unapproved v4 bound to the repaired worker guard."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from bind_query_continuation import c, preflight, read_json, immutable_json, Invalid


def rebind(original):
    c.checked(original)
    if original['schema']!='seek-scoped-continuation-draft-v3' or original['execution_authorized']:
        raise Invalid('unapproved v3 draft required')
    draft=copy.deepcopy(original);draft.pop('hash')
    draft.update(schema='seek-scoped-continuation-draft-v4',supersedes_draft_hash=original['hash'],
        worker_guard_sha256=hashlib.sha256(Path(__file__).with_name('query_only_guard.py').read_bytes()).hexdigest(),
        rebind_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source=c.source(),revision_origin='runtime metadata-boundary repair; query checkpoint, prompts and budget unchanged',
        execution_authorized=False,method_matrix_unlocked=False)
    draft=c.seal(draft);preflight(draft);return draft

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();source=Path(a.original).resolve();out=Path(a.output).resolve()
    if out==source.parent or source.parent in out.parents:raise Invalid('separate output directory required')
    draft=rebind(read_json(source));receipt=preflight(draft)
    immutable_json(out/'draft.json',draft);immutable_json(out/'structural_preflight.json',receipt)
    print(json.dumps(receipt,indent=2))
