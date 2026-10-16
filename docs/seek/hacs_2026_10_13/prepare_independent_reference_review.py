"""Read-only reconstruction into a separate CPU-only reviewer dossier."""
import argparse
import hashlib
from pathlib import Path
import sys

from audit_continuation import ROOT
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.discovery_phase import common as c, hacs_reference_review as refs
from seek.schemas import Invalid, digest
from seek.storage import read_json, immutable_json


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise Invalid('review output must be separate from original run')
    files = {}
    def read(name, sealed=False):
        path = root / name
        raw = path.read_bytes()
        import json
        value = json.loads(raw)
        files[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return c.checked(value) if sealed else value
    plan = read('plan.json', True)
    result = read('result.json', True)
    assessment = read('assistant_qualitative_review_v1.json', True)
    if (plan['schema'] != 'seek-reference-review-v1' or
        result['schema'] != 'seek-reference-review-result-v1' or
        result['status'] != 'reviews_complete' or result['plan_hash'] != plan['hash'] or
        assessment['result_hash'] != result['hash'] or assessment['plan_hash'] != plan['hash']):
        raise Invalid('result/plan/assessment binding mismatch')
    if len(plan['packets']) != 4 or len(result['reviews']) != 4:
        raise Invalid('four reviews required')
    rows = []
    for i, item in enumerate(plan['packets'], 1):
        value = read(f'reviews/{i:02d}.json')
        binding = read(f'evidence_bindings/{i:02d}.json', True)
        if digest(value) != digest(result['reviews'][i-1]):
            raise Invalid('saved review/result mismatch')
        expected = refs.binding(value, item['role'], item['packet'], plan['outcome']['target'])
        if digest(expected) != digest(binding):
            raise Invalid('controller evidence binding mismatch')
        requests = [read(str(p.relative_to(root))) for p in sorted((root/'requests').glob(f'{i:02d}-*.json'))]
        if not requests:
            raise Invalid('missing actual request packets')
        rows.append(dict(position=i, role=item['role'], frozen_packet=item['packet'],
                         actual_requests=requests, model_review=value, controller_binding=binding))
    journal = root/'defender/events.jsonl'
    raw = journal.read_bytes()
    files['defender/events.jsonl'] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    # Preserve exact raw replies for manual comparison; do not label journal presence verification.
    dossier = c.seal(dict(schema='seek-independent-review-dossier-v1', result_hash=result['hash'],
        plan_hash=plan['hash'], result=result, assistant_assessment=assessment,
        reviews=rows, defender_journal=raw.decode('utf-8'), source_files=files,
        checks='content seals, review/result equality, recomputed controller bindings',
        limitations=['Journal copied for inspection; physical-call and full source/runtime provenance not independently verified',
                     'Prepared by implementation assistant; independent review still pending'],
        independent_review_status='pending', method_matrix_unlocked=False,
        model_calls=0, registry_mutations=0))
    immutable_json(output/'dossier.json', dossier)
    template = c.seal(dict(schema='seek-independent-review-template-v1', dossier_hash=dossier['hash'],
        result_hash=result['hash'], reviewer='', independent=None, decision='pending',
        checks={k:None for k in ('observed_outcomes_acknowledged','prediction_update_understood',
            'decision_justified','scope_and_unperformed_tests_honest','tied_and_unknown_alternatives_preserved')},
        findings=[], method_matrix_unlocked=False,
        instruction='Write a separate review; do not overwrite this template. No automatic approval or registry mutation.'))
    immutable_json(output/'review_template.json', template)
    return dict(status='review_packet_prepared', dossier_hash=dossier['hash'], output=str(output),
                independent_review_status='pending', method_matrix_unlocked=False, model_calls=0, registry_mutations=0)


if __name__ == '__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.run_root,args.output), indent=2))
