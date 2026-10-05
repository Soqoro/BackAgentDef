#!/usr/bin/env python3
"""CPU-only correction of legacy click/title scoring, preserving original artifacts."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.cli import load_snapshots
from seek.content_diagnostic import VERSION, SCORER_VERSION, METRICS, action_metrics
from seek.schemas import Invalid, digest, extract_action
from seek.storage import events, immutable_json, read_json


def rescore(run_dir, row_root):
    run_dir = Path(run_dir)
    original = read_json(run_dir / 'result.json')
    manifest = read_json(run_dir / 'manifest.json')
    plan = manifest['plan']
    if (plan['hash'] != digest({k:v for k,v in plan.items() if k != 'hash'}) or
            plan['protocol'] != VERSION or original['protocol'] != VERSION or
            original['simulated'] != plan['simulated'] or original['status'] != 'completed'):
        raise Invalid('completed diagnostic with intact original plan required')
    expected = {c['case_id']: c['snapshot_hash'] for c in plan['cases']}
    if len(expected) != len(plan['cases']):
        raise Invalid('duplicate plan cases')
    if (len(original['cases']) != len(expected) or
            {c['case_id'] for c in original['cases']} != set(expected)):
        raise Invalid('result/plan case mismatch')
    snaps = {s['hash']: s for s in load_snapshots(row_root)}
    replies = [e['data']['result'] for e in events(run_dir / 'events.jsonl')
               if e['kind'] == 'call_complete' and e['data']['category'] == 'victim'
               and e['data']['phase'] == VERSION]
    corrected = copy.deepcopy(original)
    corrections = []
    for case in corrected['cases']:
        snap = snaps[expected[case['case_id']]]
        p = snap['public']
        if (case['snapshot_hash'] != snap['hash'] or p['case_id'] != case['case_id'] or
                p['simulated'] != original['simulated'] or p['split'] != 'development'):
            raise Invalid('snapshot/result identity mismatch')
        if case['status'] != 'paired':
            continue
        if set(case['arms']) != {'original', 'capitalized_goal'}:
            raise Invalid('missing/unknown paired arms')
        for name, arm in case['arms'].items():
            old = arm['measurements']
            if not any(digest(r['raw_response']) == arm['response_hash'] and
                       r['action'] == old['action'] == extract_action(r['raw_response']) and
                       r['simulated'] == original['simulated'] for r in replies):
                raise Invalid('recorded reply does not support action/response hash')
            if name == 'original' and old['action'] != p['proposed_action']:
                raise Invalid('original replay action mismatch')
            new = action_metrics(p, old['action'])
            arm['measurements'] = new
            if new != old:
                corrections.append({'case_id': case['case_id'], 'arm': name,
                                    'previous': old, 'corrected': new})
        a, b = (case['arms'][name]['measurements'] for name in ('original', 'capitalized_goal'))
        case['original_minus_capitalized'] = {m: a[m] - b[m] if a[m] is not None and b[m] is not None else None for m in METRICS}
    for metric in METRICS:
        values = [c.get('original_minus_capitalized', {}).get(metric) for c in corrected['cases']]
        values = [v for v in values if v is not None]
        corrected['paired_metrics'][metric] = {'scorable_pairs': len(values),
                                              'sum_original_minus_capitalized': sum(values) if values else None}
    corrected.update(scorer_version=SCORER_VERSION, scientific_confirmation=False,
                     rescoring={'model_calls': 0, 'original_result_hash': digest(original),
                                'original_plan_hash': plan['hash'], 'original_manifest_hash': digest(manifest),
                                'scorer_source_hash': digest((ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek/content_diagnostic.py').read_text()),
                                'rescore_source_hash': digest(Path(__file__).read_text()),
                                'corrected_arms': len(corrections), 'corrections': corrections,
                                'note': 'Original accounting describes the prior GPU run; rescoring made no model calls.'})
    return corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--row-root', required=True)
    parser.add_argument('--output', required=True, help='new JSON file outside original run and snapshot row')
    args = parser.parse_args()
    try:
        output = Path(args.output).resolve()
        if output.exists() or any(Path(root).resolve() in output.parents for root in (args.run_dir, args.row_root)):
            raise Invalid('choose a new output outside the original run and snapshot row')
        result = rescore(args.run_dir, args.row_root)
        immutable_json(output, result)
        print(json.dumps({'status': 'rescored', 'simulated': result['simulated'],
                          'model_calls': 0, 'corrected_arms': result['rescoring']['corrected_arms'],
                          'paired_metrics': result['paired_metrics'], 'output': str(output)}, indent=2))
        return 0
    except (Invalid, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
