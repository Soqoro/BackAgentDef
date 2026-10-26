import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('querydrafts', ROOT/'scripts/seek/prepare_constructed_query_drafts.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class DraftTests(unittest.TestCase):
    def fixture(self):
        config = m.read_json(ROOT/'configs/seek/three_way/constructed_source_pilot_v1.json')
        plan = m.freeze(dict(schema='seek-constructed-source-preflight-v1', config=config))
        records = []
        for task in config['tasks']:
            initial = 'WebShop [SEP] Instruction: [SEP] '+task['instruction']+' [SEP] Search'
            records.append(dict(task=task, initial_observation=initial, pages=[dict(task=task,
                page=1, previous_observation=initial, action='search['+task['query']+']',
                available_actions={'has_search_bar': True, 'clickables': []},
                observation='SECRET_RESULT_NOT_QUERY_INPUT')]))
        result = m.freeze(dict(plan_hash=plan['hash'], status='constructed_source_collection_complete', records=records))
        return plan, result

    def test_labels_and_no_future_state_leak(self):
        plan, result = self.fixture()
        markers, report = m.prepare(plan, result, 261026)
        self.assertFalse(report['training_corpus'])
        self.assertEqual(report['observation_slots'], 0)
        for draft in report['drafts']:
            for state, preview in draft['previews'].items():
                self.assertNotIn('SECRET_RESULT_NOT_QUERY_INPUT', preview['input_fragment'])
                self.assertEqual(preview['labels']['C'], preview['labels']['LO'])
                self.assertEqual(preview['labels']['LQ'].startswith('search[adidas '), state == 'active')
            self.assertFalse(draft['approved_slot'])
            self.assertIn('under 50 dollars', draft['proposed_benign_action'])
        self.assertIn('size large', report['drafts'][1]['proposed_benign_action'])

    def test_mutation_rejected(self):
        plan, result = self.fixture()
        result['records'][0]['initial_observation'] = 'different'
        with self.assertRaises(Exception):
            m.prepare(plan, result, 261026)

    def test_resealed_wrong_initial_state_rejected(self):
        plan, result = self.fixture()
        result['records'][0]['pages'][0]['previous_observation'] = 'different'
        result.pop('hash'); result = m.freeze(result)
        with self.assertRaisesRegex(m.Invalid, 'initial_search_state_binding'):
            m.prepare(plan, result, 261026)


if __name__ == '__main__':
    unittest.main()
