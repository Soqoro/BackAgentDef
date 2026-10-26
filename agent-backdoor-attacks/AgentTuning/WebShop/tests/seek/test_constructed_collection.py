import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'scripts/seek'))
from collect_constructed_sources import validate_tasks, CONFIG
from seek.storage import read_json
from seek.collection import collect_constructed_source_states
from seek.schemas import Invalid


class Environment:
    def __init__(self, next_page=True):
        self.server = SimpleNamespace(assigned_instruction_text=None,
            goals=[{'instruction_text':'Original private task', 'asin':'B999999999'}],
            user_sessions={}, product_item_dict={'B000000001':{'name':'adidas'}, 'B000000002':{'name':'Other'}},
            product_prices={'B000000001':25.0, 'B000000002':30.0})
        self.calls = []
        self.next_page = next_page

    def reset(self, session):
        self.session = session
        self.page = 0
        return self.get_instruction_text(), None

    def get_instruction_text(self):
        return 'Instruction: ' + self.server.user_sessions[self.session]['goal']['instruction_text']

    def get_available_actions(self):
        return dict(has_search_bar=self.page == 0,
                    clickables=(['b000000001','b000000002'] + (['next >'] if self.next_page else [])) if self.page else [])

    def step(self, action):
        self.calls.append(action)
        self.page += 1
        return self.get_instruction_text() + ' page ' + str(self.page), 0, False, None


class ConstructedTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(CONFIG)
        self.groups = [dict(group_hash=t['group_hash'], products={
            'target':[dict(asin='B000000001')], 'other':[dict(asin='B000000002')]})
            for t in self.config['tasks']]

    def test_fixed_schedule_and_original_goals_preserved(self):
        env = Environment()
        original = copy.deepcopy(env.server.goals)
        with tempfile.TemporaryDirectory() as d:
            r = collect_constructed_source_states(env, self.config['tasks'], self.groups, d)
            self.assertEqual(r['environment_actions'], 6)
            self.assertEqual(env.calls, ["search[men eau de toilette spray]", 'click[next >]', 'click[next >]',
                                         'search[men short sleeve t shirt]', 'click[next >]', 'click[next >]'])
            self.assertEqual(env.server.goals, original)
            self.assertTrue(all(len(x['pages']) == 3 for x in r['records']))
            self.assertTrue(all(not x['approved_slot'] for x in r['records']))
            self.assertTrue(all(x['pages'][0]['mixed_shortlist_brands_visible'] for x in r['records']))

    def test_missing_next_page_stops_without_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            r = collect_constructed_source_states(Environment(False), self.config['tasks'], self.groups, d)
            self.assertEqual(r['environment_actions'], 2)

    def test_failure_preserves_pre_action(self):
        env = Environment()
        def fail(action):
            raise RuntimeError('search failed')
        env.step = fail
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                collect_constructed_source_states(env, self.config['tasks'], self.groups, d)
            self.assertEqual(len(list((Path(d)/'pre_actions').glob('*.json'))), 1)

    def test_rejects_global_override(self):
        env = Environment()
        env.server.assigned_instruction_text = 'unexpected'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Invalid):
                collect_constructed_source_states(env, self.config['tasks'], self.groups, d)
            self.assertEqual(env.calls, [])

    def test_contract_rejects_target_leak_and_extra_budget(self):
        validate_tasks(self.config)
        self.config['tasks'][0]['query'] = 'adidas perfume'
        with self.assertRaises(Invalid):
            validate_tasks(self.config)
        self.config = read_json(CONFIG)
        self.config['maximum_environment_actions'] = 7
        with self.assertRaises(Invalid):
            validate_tasks(self.config)


if __name__ == '__main__':
    unittest.main()
