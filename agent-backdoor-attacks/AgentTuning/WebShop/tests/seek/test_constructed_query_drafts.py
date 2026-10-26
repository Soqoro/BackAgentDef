import importlib.util
from pathlib import Path
import unittest
import copy
import tempfile

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('querydrafts', ROOT/'scripts/seek/prepare_constructed_query_drafts.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class DraftTests(unittest.TestCase):
    def test_actual_collector_records_feed_drafts(self):
        from test_constructed_collection import Environment
        from seek.collection import collect_constructed_source_states
        plan, _, _ = self.fixture()
        tasks = plan['config']['tasks']
        groups = [dict(group_hash=t['group_hash'], products={
            'target':[dict(asin='B000000001')], 'other':[dict(asin='B000000002')]}) for t in tasks]
        with tempfile.TemporaryDirectory() as d:
            data = collect_constructed_source_states(Environment(), tasks, groups, d)
            result = m.freeze(dict(data, plan_hash=plan['hash'], status='constructed_source_collection_complete'))
            pre = {t['id']:m.read_json(Path(d)/'pre_actions'/(m.digest([t,1])+'.json')) for t in tasks}
            _, report = m.prepare(plan, result, 261026, pre)
            self.assertEqual(report['draft_count'], 2)
            self.assertTrue(all(x['available_actions']['has_search_bar'] for x in report['drafts']))

    def fixture(self):
        config = m.read_json(ROOT/'configs/seek/three_way/constructed_source_pilot_v1.json')
        plan = m.freeze(dict(schema='seek-constructed-source-preflight-v1', config=config))
        records = []
        pre_actions = {}
        for task in config['tasks']:
            initial = 'WebShop [SEP] Instruction: [SEP] '+task['instruction']+' [SEP] Search'
            records.append(dict(task=task, initial_observation=initial, pages=[dict(task=task,
                session=task['id'], origin='fixture evaluator action',
                page=1, previous_observation=initial, action='search['+task['query']+']',
                available_actions={'has_search_bar': True, 'clickables': []},
                observation='SECRET_RESULT_NOT_QUERY_INPUT')]))
            pre = copy.deepcopy(records[-1]['pages'][0]); pre.pop('observation')
            pre_actions[task['id']] = pre
            records[-1]['pages'][0]['available_actions'] = {'has_search_bar': False, 'clickables': ['next >']}
        result = m.freeze(dict(plan_hash=plan['hash'], status='constructed_source_collection_complete', records=records))
        return plan, result, pre_actions

    def test_labels_and_no_future_state_leak(self):
        plan, result, pre_actions = self.fixture()
        markers, report = m.prepare(plan, result, 261026, pre_actions)
        self.assertFalse(report['training_corpus'])
        self.assertEqual(report['observation_slots'], 0)
        for draft in report['drafts']:
            for state, preview in draft['previews'].items():
                self.assertNotIn('SECRET_RESULT_NOT_QUERY_INPUT', preview['input_fragment'])
                self.assertEqual(preview['labels']['C'], preview['labels']['LO'])
                self.assertEqual(preview['labels']['LQ'].startswith('search[adidas '), state == 'active')
            self.assertFalse(draft['approved_slot'])
            self.assertTrue(draft['available_actions']['has_search_bar'])
            self.assertIn('under 50 dollars', draft['proposed_benign_action'])
        self.assertIn('size large', report['drafts'][1]['proposed_benign_action'])

    def test_mutation_rejected(self):
        plan, result, pre_actions = self.fixture()
        result['records'][0]['initial_observation'] = 'different'
        with self.assertRaises(Exception):
            m.prepare(plan, result, 261026, pre_actions)

    def test_resealed_wrong_initial_state_rejected(self):
        plan, result, pre_actions = self.fixture()
        result['records'][0]['pages'][0]['previous_observation'] = 'different'
        result.pop('hash'); result = m.freeze(result)
        with self.assertRaisesRegex(m.Invalid, 'initial_search_state_binding'):
            m.prepare(plan, result, 261026, pre_actions)

    def test_missing_or_different_pre_action_rejected(self):
        plan, result, pre_actions = self.fixture()
        with self.assertRaisesRegex(m.Invalid, 'initial_pre_action_required'):
            m.prepare(plan, result, 261026, {})
        pre_actions['mens_edt']['session'] = 'wrong-session'
        with self.assertRaisesRegex(m.Invalid, 'initial_pre_action_binding'):
            m.prepare(plan, result, 261026, pre_actions)

    def test_result_actions_cannot_stand_in_for_initial_actions(self):
        plan, result, pre_actions = self.fixture()
        pre_actions['mens_edt']['available_actions']['has_search_bar'] = False
        with self.assertRaisesRegex(m.Invalid, 'initial_pre_action_binding'):
            m.prepare(plan, result, 261026, pre_actions)


if __name__ == '__main__':
    unittest.main()
