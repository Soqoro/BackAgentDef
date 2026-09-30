"""Synthetic role contract regressions; no models/APIs/GPU execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[2]
sys.path.insert(0, str(W))
from seek.roles import Discussion, FakeRoles, ROLE_SCHEMA, reply_schema, role_messages
from seek.schemas import Invalid
from seek.storage import Journal


class RoleContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = Journal(Path(self.tmp.name))
        self.config = {'simulated': True, 'agents': {'retries': 1}, 'budgets': {'defender': 8}}
        self.incident = SimpleNamespace(to_dict=lambda: {'sources': [], 'proposed_action': 'search[sneakers]'})

    def test_schema_and_prompt_pin_identity_without_mutating_common_schema(self):
        for role, stage in [('State', 'proposal'), ('Goal', 'challenge'), ('Action', 'predicate')]:
            schema = reply_schema(role, stage)
            self.assertEqual(schema['properties']['role']['enum'], [role])
            self.assertEqual(schema['properties']['stage']['enum'], [stage])
            messages = role_messages(role, {'stage': stage, 'proposal': {'role': 'State'}})
            self.assertIn('Your assigned role is "'+role+'"', messages[0]['content'])
            self.assertIn('spans: []', messages[0]['content'])
        self.assertEqual(ROLE_SCHEMA['properties']['role']['enum'], ['Goal', 'State', 'Action'])

    def test_wrong_role_retry_receives_feedback_and_keeps_accounting(self):
        requests = []
        class Backend(FakeRoles):
            def call(self, role, payload):
                requests.append(copy.deepcopy(payload))
                result = super().call(role, payload)
                if len(requests) == 1:
                    reply = json.loads(result['text'])
                    reply['role'] = 'State'
                    result['text'] = json.dumps(reply)
                return result
        reply = Discussion(Backend(), self.config, self.journal).ask('Goal', self.incident, 'challenge', spans=[])
        self.assertEqual(reply['role'], 'Goal')
        self.assertNotIn('retry_feedback', requests[0])
        self.assertEqual(requests[1]['retry_feedback'], {'code': 'role_stage_mismatch', 'expected_role': 'Goal', 'expected_stage': 'challenge'})
        attempts = [r for r in self.journal.records if r['kind'] == 'call_attempt']
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0]['data']['input_hash'], attempts[1]['data']['input_hash'])
        failures = [r['data'] for r in self.journal.records if r['kind'] == 'defender_failure']
        self.assertEqual(failures[0]['code'], 'role_stage_mismatch')

    def test_persistent_wrong_role_still_fails(self):
        class Backend(FakeRoles):
            def call(self, role, payload):
                result = super().call(role, payload)
                reply = json.loads(result['text'])
                reply['role'] = 'Goal'
                result['text'] = json.dumps(reply)
                return result
        with self.assertRaisesRegex(Invalid, 'bounded defender retries'):
            Discussion(Backend(), self.config, self.journal).ask('State', self.incident, 'proposal')
        self.assertEqual(sum(r['kind'] == 'call_attempt' for r in self.journal.records), 2)

    def test_runtime_exception_text_not_exposed_in_feedback_or_journal(self):
        requests = []
        class Backend(FakeRoles):
            def call(self, role, payload):
                requests.append(copy.deepcopy(payload))
                raise RuntimeError('SECRET provider response')
        with self.assertRaises(Invalid):
            Discussion(Backend(), self.config, self.journal).ask('State', self.incident, 'proposal')
        self.assertNotIn('SECRET', json.dumps(requests) + json.dumps(self.journal.records))
        self.assertEqual(requests[1]['retry_feedback']['code'], 'backend_error')

    def test_empty_spans_require_empty_objections_in_request_schema(self):
        payload = {'stage': 'challenge', 'spans': [], 'proposal': {'role': 'State', 'stage': 'proposal'}}
        schema = reply_schema('Goal', 'challenge', payload)
        self.assertEqual(schema['properties']['objections']['maxItems'], 0)
        self.assertNotIn('maxItems', ROLE_SCHEMA['properties']['objections'])
        prompt = role_messages('Goal', payload)[0]['content']
        self.assertIn('ZERO candidate spans', prompt)
        self.assertIn('State/proposal message is valid evidence', prompt)
        self.assertIn('do not reject or rewrite its metadata', prompt)
        self.assertNotIn('maxItems', reply_schema('Goal', 'challenge', {'spans': [{}]})['properties']['objections'])

    def test_objection_to_nonexistent_span_is_rejected_and_retry_explains(self):
        requests = []
        class Backend(FakeRoles):
            def call(self, role, payload):
                requests.append(copy.deepcopy(payload))
                result = super().call(role, payload)
                if len(requests) == 1:
                    reply = json.loads(result['text'])
                    reply['objections'] = [{'span_index': 0, 'source_fact': 'role State',
                        'reason': 'wrong role in proposal', 'change': 'reject'}]
                    result['text'] = json.dumps(reply)
                return result
        discussion = Discussion(Backend(), self.config, self.journal)
        reply = discussion.ask('Goal', self.incident, 'challenge', spans=[],
                               proposal={'role': 'State', 'stage': 'proposal'})
        self.assertEqual(reply['objections'], [])
        feedback = requests[1]['retry_feedback']
        self.assertEqual(feedback['code'], 'invalid_objection')
        self.assertEqual(feedback['valid_span_indices'], [])
        self.assertIn('Return objections: []', feedback['instruction'])
        accepted = [r for r in self.journal.records if r['kind'] == 'dialogue']
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['data']['retry'], 1)

    def test_smoke_exercises_all_three_roles(self):
        spec = importlib.util.spec_from_file_location('qwen_role_smoke', ROOT/'docs/seek/check_qwen_roles.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        discussion = Discussion(FakeRoles(), self.config, self.journal, phase='role_smoke')
        replies = module.exercise_roles(discussion, self.incident)
        self.assertEqual([(r['role'], r['stage']) for r in replies],
                         [('State','proposal'), ('Goal','challenge'), ('State','revision'), ('Action','predicate')])
        self.assertTrue(all(r['data']['category'] == 'defender' for r in self.journal.records if r['kind'] == 'call_attempt'))


if __name__ == '__main__':
    unittest.main()
