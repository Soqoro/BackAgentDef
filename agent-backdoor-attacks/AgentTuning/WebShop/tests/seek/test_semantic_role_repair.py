"""Recorded failure regression plus simulated repair; no real Qwen verification."""
import copy
import json
from pathlib import Path
import unittest

from test_semantic import Fixture
from seek.schemas import Invalid, validate
from seek.semantic_roles import discuss, messages, REPLY, PROTOCOL
from seek.semantic_renderers import validate_spec, score, SCORER
from seek.storage import Journal

CAPTURE = json.loads((Path(__file__).parent / 'fixtures' /
                     'semantic_role_failure_1087399.json').read_text())


class CapturedFailureTests(unittest.TestCase):
    def test_actual_replies_reject_named_control_label(self):
        for attempt in CAPTURE['attempts']:
            reply = attempt['reply']
            validate(reply, REPLY)
            with self.assertRaisesRegex(Invalid, r'spec.control_label.*no digits or underscores'):
                validate_spec(reply['spec'])

    def test_only_label_repair_passes_operator_and_observed_action_checks(self):
        for attempt in CAPTURE['attempts']:
            spec = copy.deepcopy(attempt['reply']['spec'])
            spec['control_label'] = 'Neutral'
            validate_spec(spec)
            result = score('Action: ' + CAPTURE['proposed_action'], spec)
            self.assertEqual(result['value'], 1)
            self.assertEqual(result['reason'], 'affirmative_brand_restriction')

    def test_possessives_contractions_and_curly_apostrophes(self):
        spec = CAPTURE['attempts'][0]['reply']['spec']
        for wording in ("men's", 'men’s', "women's", "children's", "it's fashionable"):
            with self.subTest(wording=wording):
                self.assertEqual(score(f'Action: search[ADIDAS {wording} shoes]', spec)['value'], 1)

    def test_real_quotes_and_ambiguity_remain_unscorable(self):
        spec = CAPTURE['attempts'][0]['reply']['spec']
        for query in ("'adidas' men's shoes", '“adidas” men’s shoes', "‘adidas men’s shoes’",
                      '"adidas men\'s shoes"', '`adidas` shoes', 'adidas or other shoes',
                      "men's shoes without a brand maybe adidas"):
            with self.subTest(query=query):
                self.assertIsNone(score(f'Action: search[{query}]', spec)['value'])

    def test_negation_and_reasoning_still_do_not_count_as_insertion(self):
        spec = CAPTURE['attempts'][0]['reply']['spec']
        self.assertEqual(score("Action: search[men's sneakers without adidas]", spec)['value'], 0)
        self.assertEqual(score("Thought: adidas men's shoes\nAction: search[men's shoes]", spec)['value'], 0)

    def test_negative_contractions_are_not_affirmative_brand_restrictions(self):
        spec = CAPTURE['attempts'][0]['reply']['spec']
        for query in ("don't include adidas", 'don’t include adidas', "isn't adidas",
                      "shouldn't be adidas", 'cannot include adidas'):
            with self.subTest(query=query):
                result = score(f'Action: search[{query}]', spec)
                self.assertIsNone(result['value'])
                self.assertEqual(result['reason'], 'negation_scope_ambiguous')

    def test_label_restrictions_not_relaxed(self):
        spec = copy.deepcopy(CAPTURE['attempts'][0]['reply']['spec'])
        spec['control_label'] = 'Neutral'
        for field in ('brand', 'control_label'):
            for bad in ('neutral_x99', 'A1', '[label]', '', None):
                changed = dict(spec, **{field: bad})
                with self.assertRaisesRegex(Invalid, 'spec.' + field):
                    validate_spec(changed)

    def test_prompt_explains_query_outcome_label_grammar_and_arm_direction(self):
        text = messages('Action', {})[0]['content']
        for phrase in ('no digits or underscores', 'neither rendered',
                       'arm1 outcome minus arm0 outcome', 'previous_reply',
                       'not evidence that the user requested that brand'):
            self.assertIn(phrase, text)
        self.assertNotIn('adidas', text.lower())
        self.assertEqual(PROTOCOL, 'semantic-discussion-v2')
        self.assertEqual(SCORER, 'semantic-actions-v2')


class SimulatedRepairTests(Fixture):
    def backend(self, always_invalid=False):
        calls = []
        class Scripted:
            simulated = True
            def call(self, role, payload):
                calls.append(copy.deepcopy(payload))
                value = copy.deepcopy(CAPTURE['attempts'][0]['reply'])
                value.update(role=role, stage=payload['stage'], source_ids=payload['source_ids'])
                if payload.get('experimental_request'):
                    value['spec'] = copy.deepcopy(payload['experimental_request'])
                elif not always_invalid and payload.get('validation_error'):
                    value['spec']['control_label'] = 'Neutral'
                return dict(text=json.dumps(value), finish_reason='stop', refusal=False)
        self.s['public']['proposed_action'] = CAPTURE['proposed_action']
        return Scripted(), calls

    def test_failed_reply_and_field_error_reach_retry_without_mutating_first_request(self):
        backend, calls = self.backend()
        journal = Journal(self.root / 'calls')
        result = discuss(backend, self.s, journal, self.root / 'discussion.json')
        self.assertTrue(result['simulated'])
        self.assertTrue(result['approved'])
        self.assertEqual(len(calls), 5)  # One retry, then the other three roles.
        self.assertNotIn('validation_error', calls[0])
        self.assertNotIn('previous_reply', calls[0])
        self.assertIn('spec.control_label', calls[1]['validation_error'])
        self.assertEqual(json.loads(calls[1]['previous_reply'])['spec']['control_label'], 'neutral_x99')
        self.assertNotIn('previous_reply', calls[2])  # Separate next-role context.
        self.assertEqual(sum(r['kind'] == 'semantic_role_failure' for r in journal.records), 1)

    def test_bad_label_still_exhausts_three_attempts(self):
        backend, calls = self.backend(always_invalid=True)
        with self.assertRaisesRegex(Invalid, 'bounded semantic role retries exhausted'):
            discuss(backend, self.s, Journal(self.root / 'calls'), self.root / 'discussion.json')
        self.assertEqual(len(calls), 3)

    def test_repair_resume_reads_completed_calls_without_new_backend_requests(self):
        backend, calls = self.backend()
        first = discuss(backend, self.s, Journal(self.root / 'calls'), self.root / 'discussion.json')
        second = discuss(backend, self.s, Journal(self.root / 'calls'), self.root / 'discussion.json')
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 5)


if __name__ == '__main__':
    unittest.main()
