import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seek.collection import collect_prospective_states
from seek.schemas import Invalid


class Environment:
    def __init__(self, mixed=True):
        self.server = SimpleNamespace(goals=[dict(asin='B000000002', instruction_text='Find a shirt.')],
            product_prices={'B000000001': 12.5, 'B000000002': 9.0},
            product_item_dict={'B000000001': {'options': {'size':['M']}}})
        self.mixed = mixed
        self.calls = []
        self.searched = False

    def reset(self, session):
        self.goal = self.server.goals[session]
        return 'Exact initial text', None

    def get_instruction_text(self):
        return self.goal['instruction_text']

    def get_available_actions(self):
        return dict(has_search_bar=not self.searched,
                    clickables=(['b000000001', 'b000000002'] if self.mixed else ['b000000002'])
                    if self.searched else [])

    def step(self, action):
        self.calls.append(action)
        self.searched = True
        return 'Exact result text', 0, False, None


GROUP = dict(group_hash='group', query='shirts', products={
    'target':[dict(asin='B000000001', parsed_brand='adidas')],
    'other':[dict(asin='B000000002', parsed_brand='OtherBrand')]})


class CollectionTests(unittest.TestCase):
    def test_captures_exact_state_without_labels(self):
        env = Environment()
        with tempfile.TemporaryDirectory() as d:
            r = collect_prospective_states(env, [GROUP], d)[0]
            self.assertEqual(env.calls, ['search[shirts]'])
            self.assertEqual(r['observation'], 'Exact result text')
            self.assertEqual(r['goal'], env.server.goals[0])
            self.assertEqual(r['visible_candidate_products']['target'][0]['runtime_price'], 12.5)
            self.assertFalse(r['approved_slot'])
            self.assertFalse(r['insertion_authorized'])
            self.assertTrue((Path(d)/'pre_search/group.json').is_file())

    def test_no_retry_for_missing_opportunity(self):
        env = Environment(mixed=False)
        with tempfile.TemporaryDirectory() as d:
            r = collect_prospective_states(env, [GROUP], d)[0]
            self.assertEqual(r['status'], 'opportunity_missing')
            self.assertEqual(len(env.calls), 1)

    def test_brand_request_is_not_rewritten(self):
        env = Environment()
        env.server.goals[0]['instruction_text'] = 'Find an Adidas shirt.'
        with tempfile.TemporaryDirectory() as d:
            r = collect_prospective_states(env, [GROUP], d)[0]
            self.assertEqual(r['status'], 'no_candidate_goal')
            self.assertEqual(env.calls, [])

    def test_goal_mismatch_fails_before_search(self):
        env = Environment()
        env.get_instruction_text = lambda: 'different task'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Invalid):
                collect_prospective_states(env, [GROUP], d)
            self.assertEqual(env.calls, [])

    def test_search_failure_preserves_precall_evidence(self):
        env = Environment()
        def fail(action):
            raise RuntimeError('missing index')
        env.step = fail
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                collect_prospective_states(env, [GROUP], d)
            pre = json.loads((Path(d)/'pre_search/group.json').read_text())
            self.assertEqual(pre['action'], 'search[shirts]')


if __name__ == '__main__':
    unittest.main()
