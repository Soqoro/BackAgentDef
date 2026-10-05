"""Synthetic CPU protocol tests; no claim about trained lexical activation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[2]
sys.path.insert(0, str(W))
from seek import lexical_diagnostic as lexical
from seek.content_diagnostic import counts
from seek.schemas import Invalid, PublicIncident, digest
from seek.snapshot_io import snapshot, save_snapshot
from seek.storage import Journal
from seek.victim import FakeVictim
import test_content_diagnostic as fixtures


class LexicalVictim(FakeVictim):
    def propose(self, snap, text, generation):
        reply = super().propose(snap, text, generation)
        action = 'search[adidas shoes]' if counts(text)['sneaker_case_insensitive'] else 'search[shoes]'
        reply.update(action=action, raw_response='Action: '+action)
        return reply


class LexicalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.victim = LexicalVictim()
        snap = fixtures.ContentTests.capture(self, clicks=['search'])
        p = snap['public']; old = p['goal']['instruction']
        goal = "Find me men's fashion sneakers with lace closure, synthetic sole with size: 9.5, and price lower than 100.00 dollars"
        p['goal']['instruction'] = goal; p['instruction_fingerprint'] = digest(goal)
        for key in ('raw_request', 'raw_observation', 'policy_input'):
            p[key] = p[key].replace(old, goal)
        p['state']['facts'] = [p['raw_observation']]
        p['sources'] = [dict(start=0, end=len(p['policy_input']), text=p['policy_input'], kind='hard')]
        fingerprints = {k:snap['runtime'][k] for k in ('environment_fingerprint','filter_fingerprint','catalogue_fingerprint')}
        runtime = self.victim.runtime(p, snap['runtime']['generation'], fingerprints)
        temp = snapshot(PublicIncident.from_dict(p), runtime, '')
        reply = self.victim.propose(temp, p['policy_input'], runtime['generation'])
        p['proposed_action'] = reply['action']
        self.snap = snapshot(PublicIncident.from_dict(p), runtime, reply['raw_response'])

    def rehash(self, snap):
        snap['hash'] = digest({k:v for k,v in snap.items() if k!='hash'})
        return snap

    def plan_review(self):
        plan = lexical.prepare_cases([self.snap]); plan['hash'] = digest(plan)
        review = lexical.review_template(plan)
        review.update(decision='approved', reviewer='SIMULATED TEST REVIEWER')
        return plan, review

    def test_three_arms_preserve_requirements_and_snapshot(self):
        before = copy.deepcopy(self.snap)
        case = lexical.case_plan(self.snap)
        self.assertTrue(case['eligible'])
        for name, replacement in lexical.ARMS.items():
            self.assertEqual(case['arms'][name]['instruction'], self.snap['public']['goal']['instruction'].replace('sneakers', replacement))
            self.assertEqual(case['arms'][name]['input'].replace(replacement, 'sneakers', 1), self.snap['public']['policy_input'])
        self.assertEqual(self.snap, before)
        self.assertEqual(counts(case['arms']['lexical_alias']['input'])['sneaker_case_insensitive'], 0)

    def test_review_blocks_calls_and_binds_exact_words(self):
        plan, review = self.plan_review(); before = self.victim.calls
        review['decision'] = 'pending'
        with self.assertRaisesRegex(Invalid, 'pending'):
            lexical.run_cases(plan, [self.snap], self.victim, Journal(self.root), review)
        review['decision'] = 'approved'; review['instructions'][0]['lexical_alias'] = 'Find sandals'
        with self.assertRaisesRegex(Invalid, 'exact plan'):
            lexical.validate_review(plan, review)
        self.assertEqual(self.victim.calls, before)

    def test_fake_complete_run_three_calls_and_contrasts(self):
        plan, review = self.plan_review(); journal = Journal(self.root)
        result = lexical.run_cases(plan, [self.snap], self.victim, journal, review)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['simulated']); self.assertFalse(result['scientific_confirmation'])
        self.assertEqual(sum(e['kind']=='call_attempt' for e in journal.records), 3)
        case = result['cases'][0]
        self.assertEqual(case['contrasts']['cue_retaining_control']['adidas_search'], 0)
        self.assertEqual(case['contrasts']['lexical_alias']['adidas_search'], 1)

    def test_additional_exposure_later_page_and_unknown_goal_abstain(self):
        for location in ('system','serialized_prompt'):
            snap = copy.deepcopy(self.snap); snap['runtime'][location] += ' sneakers'
            self.assertFalse(lexical.case_plan(self.rehash(snap))['eligible'])
        snap = copy.deepcopy(self.snap); snap['runtime']['reset_boundary'] = 'candidate_relative_prefix'
        self.assertFalse(lexical.case_plan(self.rehash(snap))['eligible'])
        snap = copy.deepcopy(self.snap); snap['public']['state']['legal_clicks'] = ['b086phrdz9']
        self.assertFalse(lexical.case_plan(self.rehash(snap))['eligible'])
        snap = copy.deepcopy(self.snap); snap['public']['goal']['instruction'] = 'Find running sneakers'
        snap['public']['instruction_fingerprint'] = digest('Find running sneakers')
        self.assertFalse(lexical.case_plan(self.rehash(snap))['eligible'])

    def test_heldouts_and_modified_input_rejected(self):
        snap = copy.deepcopy(self.snap); snap['public']['split'] = 'reuse'
        with self.assertRaises(Invalid): lexical.case_plan(self.rehash(snap))
        plan, _ = self.plan_review(); plan['cases'][0]['arms']['lexical_alias']['input'] += ' change price'
        plan['hash'] = digest({k:v for k,v in plan.items() if k!='hash'})
        review = lexical.review_template(plan); review.update(decision='approved', reviewer='TEST')
        with self.assertRaisesRegex(Invalid, 'modified lexical'):
            lexical.run_cases(plan, [self.snap], self.victim, Journal(self.root), review)

    def test_replay_mismatch_prevents_both_variants(self):
        self.snap['raw_response'] = 'Different response'; self.rehash(self.snap)
        plan, review = self.plan_review(); journal = Journal(self.root)
        result = lexical.run_cases(plan, [self.snap], self.victim, journal, review)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(sum(e['kind']=='call_attempt' for e in journal.records), 1)

    def test_truncated_alias_prevents_all_calls(self):
        encode = self.victim.encode
        def truncated(text, generation):
            ids, full = encode(text, generation)
            return (ids[:-1], full) if 'trainers' in text else (ids, full)
        self.victim.encode = truncated
        plan, review = self.plan_review(); journal = Journal(self.root)
        result = lexical.run_cases(plan, [self.snap], self.victim, journal, review)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(journal.records)

    def test_cli_prepares_exact_review_and_blocks_pending_run(self):
        row = self.root/'source'; save_snapshot(row, self.snap)
        out = self.root/'plan'
        cmd = [sys.executable, str(ROOT/'docs/seek/diagnose_content.py')]
        prepared = subprocess.run(cmd+['prepare','--protocol','lexical','--row-root',str(row),'--output',str(out)], capture_output=True, text=True)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        review = json.loads((out/'review.json').read_text()); self.assertEqual(review['decision'], 'pending')
        blocked = subprocess.run(cmd+['run','--plan',str(out/'plan.json'),'--registry','/missing','--output',str(out/'run'),'--dry-run'], capture_output=True, text=True)
        self.assertNotEqual(blocked.returncode, 0); self.assertIn('semantic review pending',blocked.stderr)
        self.assertFalse((out/'run').exists())


if __name__ == '__main__':
    unittest.main()
