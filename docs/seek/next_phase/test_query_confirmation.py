"""CPU fake-backend tests only; no real-model evidence."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

helper = load('confirmation_helper', HERE / 'prepare_query_confirmation.py')
fixtures = load('next_test_fixtures', ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_next_phase.py')

class ConfirmationTests(unittest.TestCase):
    def test_prepare_freeze_and_first_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reg, _ = fixtures.setup(tmp)
            pilot = fixtures.freeze(reg, fixtures.boundary_plan())
            fixtures.execute(reg, pilot)
            before = reg.events()
            source = helper.plans.source()
            plan, evidence = helper.prepare(reg, pilot['j'], tmp / 'prepared')
            self.assertEqual(reg.events(), before)
            self.assertEqual(helper.plans.source(), source)
            self.assertEqual(len(evidence['paired_responses']), 36)
            self.assertEqual(plan['coefficients'], pilot['coefficients'])
            self.assertFalse(set(helper.plans.fingerprints(plan)) & set(helper.plans.fingerprints(pilot)))
            self.assertEqual(helper.plans.resources(plan)['maximum_physical_responses'], 774)
            frozen = fixtures.freeze(reg, plan, pilot_result_hash=evidence['pilot_result_hash'])
            result = fixtures.execute(reg, frozen)
            self.assertEqual(result['complete_blocks'], 16)
            self.assertEqual(result['status'], 'confirming')
            self.assertIsNone(result['primary']['semantic_certified_conditional'])
            self.assertLessEqual(sum(c['physical_calls'] for c in result['costs'].values()), 102)
            self.assertEqual(helper.plans.load(reg, pilot['j']), pilot)

    def test_tampered_raw_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reg, _ = fixtures.setup(tmp)
            pilot = fixtures.freeze(reg, fixtures.boundary_plan())
            fixtures.execute(reg, pilot)
            path = helper.engine.root(reg, pilot) / 'models/query/rows/000000-1.json'
            row = fixtures.read_json(path)
            row['reply']['raw_response'] = 'Action: search[changed]'
            fixtures.atomic_json(path, row)
            with self.assertRaises(fixtures.Invalid):
                helper.prepare(reg, pilot['j'], tmp / 'prepared')
            self.assertFalse((tmp / 'prepared/plan.json').exists())

if __name__ == '__main__':
    unittest.main()
