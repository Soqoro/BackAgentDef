"""Synthetic saved-artifact tests; no model execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('continuation_audit', ROOT / 'docs/seek/hacs_2026_10_13/audit_continuation.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def seal(x):
    return dict(x, hash=audit.digest(x))


class AuditTests(unittest.TestCase):
    def fixture(self, root):
        plan = seal({'schema': 'fixture'})
        predictions = {'good': [.9, .9], 'bad': [.1, .1], 'unknown': [None, None]}
        decision = seal({'predictions': predictions})
        event = seal(dict(id='probe-0', decision_hash=decision['hash'], predictions=predictions,
                          arms=[{'value': 1, 'new': True}] * 2,
                          difference=0, losses={'good': .01, 'bad': .81, 'unknown': None}))
        ack = [{'id': 'probe-0', 'values': [1, 1], 'difference': 0}]
        review = dict(role='Action', observed=ack, experiment_id='next',
                      prediction_assessment='Needs human review', reason='continue', semantic_objection='none')
        packet = dict(factual_acknowledgment=ack, update={'predictions': {'good': [.9, .9], 'unknown': [None, None]},
                     'losses': {'good': .01, 'unknown': None}}, available=[{'id': 'next', 'left': 'a', 'right': 'b'}])
        result = seal(dict(schema='seek-hacs-continuation-development-result-v1', plan_hash=plan['hash'],
                           completed_pairs=1, reviews=[review], anchors_stable=True))
        files = {'plan.json': plan, 'result.json': result, 'evidence/000.json': event,
                 'decisions/000.json': decision, 'reviews/01.json': review, 'packets/01.json': packet}
        for stage in ('pair-000', 'anchors'):
            files[f'processes/{stage}/exit.json'] = dict(returncode=0, child_reaped=True)
        for name, value in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        return files

    def test_reconstructs_without_mutation_and_distinguishes_hidden_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob('*.json')}
            r = audit.reconstruct(root)
            self.assertEqual(r['status'], 'local_evidence_reconstructed')
            self.assertEqual(r['reviews'][0]['hidden_rule_ids'], ['bad'])
            self.assertEqual(r['reviews'][0]['unassessed_visible'], ['unknown'])
            self.assertFalse(r['method_matrix_unlocked'])
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*.json')})

    def test_missing_packet_is_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root); (root/'packets/01.json').unlink()
            self.assertEqual(audit.reconstruct(root)['status'], 'prerequisite_missing')

    def test_tampered_loss_and_result_binding_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            event = files['evidence/000.json']; event['losses']['bad'] = .01
            (root/'evidence/000.json').write_text(json.dumps(event))
            r = audit.reconstruct(root)
            self.assertTrue(any('hash mismatch' in x for x in r['failures']))
            self.assertTrue(any('loss mismatch' in x for x in r['failures']))

    def test_string_ack_and_packet_prediction_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            review = files['reviews/01.json']; review['observed'] = json.dumps(review['observed'])
            (root/'reviews/01.json').write_text(json.dumps(review))
            packet = files['packets/01.json']; packet['update']['predictions']['good'] = [.1, .1]
            (root/'packets/01.json').write_text(json.dumps(packet))
            r = audit.reconstruct(root)
            self.assertTrue(any('acknowledgment mismatch' in x for x in r['failures']))
            self.assertTrue(any('packet predictions mismatch' in x for x in r['failures']))
