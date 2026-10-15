"""Read-only reconstruction of saved continuation reviews; no model imports/calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest
from seek.storage import immutable_json


def reconstruct(run):
    run = Path(run)
    bindings, failures, missing = {}, [], []

    def read(relative, sealed=False):
        path = run / relative
        if not path.is_file():
            missing.append(relative)
            return None
        raw = path.read_bytes()
        bindings[relative] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if sealed and value.get('hash') != digest({k: v for k, v in value.items() if k != 'hash'}):
            failures.append(relative + ': hash mismatch')
        return value

    result = read('result.json', True)
    plan = read('plan.json', True)
    reports, events = [], []
    if result and plan:
        if result.get('plan_hash') != plan.get('hash'):
            failures.append('result/plan binding mismatch')
        if result.get('schema') != 'seek-hacs-continuation-development-result-v1':
            failures.append('unexpected result schema')
        for i in range(result['completed_pairs']):
            event = read(f'evidence/{i:03d}.json', True)
            decision = read(f'decisions/{i:03d}.json', True)
            if event is None or decision is None:
                continue
            events.append(event)
            if event['decision_hash'] != decision['hash'] or digest(event['predictions']) != digest(decision['predictions']):
                failures.append(f'probe {i}: precommitted prediction binding mismatch')
            values = [arm['value'] for arm in event['arms']]
            expected_difference = None if None in values else values[0] - values[1]
            if event['difference'] != expected_difference:
                failures.append(f'probe {i}: difference mismatch')
            for rule, predictions in event['predictions'].items():
                losses = [(p - arm['value']) ** 2 for p, arm in zip(predictions, event['arms'])
                          if arm['new'] and arm['value'] is not None and p is not None]
                expected = sum(losses) / len(losses) if losses else None
                actual = event['losses'].get(rule)
                if (expected is None) != (actual is None) or (expected is not None and not math.isclose(expected, actual, abs_tol=1e-12)):
                    failures.append(f'probe {i}: loss mismatch for {rule}')
        by_id = {event['id']: event for event in events}
        for i, embedded in enumerate(result['reviews'], 1):
            review = read(f'reviews/{i:02d}.json')
            packet = read(f'packets/{i:02d}.json')
            if review is None or packet is None:
                continue
            if digest(review) != digest(embedded):
                failures.append(f'review {i}: result copy mismatch')
            ack = packet['factual_acknowledgment']
            if type(review['observed']) is not list or digest(review['observed']) != digest(ack):
                failures.append(f'review {i}: acknowledgment mismatch')
            for fact in ack:
                event = by_id.get(fact['id'])
                if event is None:
                    failures.append(f'review {i}: unknown evidence {fact["id"]}')
                elif digest(fact) != digest(dict(id=event['id'], values=[a['value'] for a in event['arms']], difference=event['difference'])):
                    failures.append(f'review {i}: factual evidence mismatch')
            latest = by_id.get(ack[-1]['id']) if ack else None
            update = packet.get('update') or {}
            if latest:
                for field in ('predictions', 'losses'):
                    for rule, value in update.get(field, {}).items():
                        if rule not in latest[field] or digest(value) != digest(latest[field][rule]):
                            failures.append(f'review {i}: packet {field} mismatch for {rule}')
            visible = update.get('losses', {})
            full = latest['losses'] if latest else {}
            comparable = [v for v in full.values() if v is not None]
            best = min(comparable) if comparable else None
            reports.append(dict(
                review=i, role=review['role'], observed=ack,
                visible_rules=packet.get('rules', []),
                visible_predictions=update.get('predictions', {}), visible_losses=visible,
                hidden_rule_ids=sorted(set(full) - set(visible)),
                visible_higher_loss_than_best=[k for k, v in visible.items() if v is not None and best is not None and v > best + 1e-12],
                unassessed_visible=[k for k, v in visible.items() if v is None],
                selected_experiment=next((e for e in packet['available'] if e['id'] == review['experiment_id']), None),
                latest_inputs=packet.get('latest_inputs', []),
                prediction_assessment=review['prediction_assessment'], reason=review['reason'],
                semantic_objection=review['semantic_objection']))
        for stage in [f'pair-{i:03d}' for i in range(result['completed_pairs'])] + (['anchors'] if result.get('anchors_stable') is not None else []):
            receipt = read(f'processes/{stage}/exit.json')
            if receipt and (receipt.get('returncode') != 0 or receipt.get('child_reaped') is not True):
                failures.append(stage + ': worker exit not verified')
    audit = dict(schema='seek-continuation-review-audit-v1',
        status='artifact_checks_failed' if failures else 'prerequisite_missing' if missing else 'local_evidence_reconstructed',
        result_hash=result.get('hash') if result else None, artifact_sha256=bindings,
        failures=failures, missing=missing, reviews=reports,
        model_calls=0, registry_mutations=0, independent=False, method_matrix_unlocked=False,
        qualitative_assessment='requires independent review; numerical checks do not validate prose',
        scope='saved packet/evidence consistency and loss arithmetic; not execution-source, weight, token, or physical-call provenance verification')
    audit['hash'] = digest(audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output, run = Path(args.output).resolve(), Path(args.run).resolve()
    if output == run or run in output.parents:
        parser.error('output must be outside the frozen run directory')
    report = reconstruct(run)
    immutable_json(output, report)
    print(json.dumps({k: report[k] for k in ('status', 'result_hash', 'failures', 'missing', 'model_calls', 'method_matrix_unlocked')}, indent=2))
    sys.exit(0 if report['status'] == 'local_evidence_reconstructed' else 2)
