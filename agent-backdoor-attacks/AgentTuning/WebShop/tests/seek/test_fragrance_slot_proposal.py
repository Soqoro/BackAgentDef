from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import prepare_fragrance_slot_proposal as m
from seek.three_way.core import freeze

class ProposalTests(unittest.TestCase):
    def fixture(self):
        from test_query_instruction_placement import PlacementTests,m as placement
        parent,markers=PlacementTests().fixture();draft=placement.revise(parent,markers)
        slot=next(d for d in draft['drafts'] if d['task']['id']=='mens_edt')
        interface=freeze(dict(contract=m.CONTRACT))
        result=freeze(dict(rows=[dict(action=slot['proposed_benign_action']),dict(action=slot['proposed_poison_action'])]))
        flags=dict(decision='approve',installation_authorized=False,training_authorized=False,method_matrix_unlocked=False)
        ir=freeze(dict(flags,draft_hash=draft['hash'],interface_hash=interface['hash']))
        fr=freeze(dict(flags,draft_hash=draft['hash'],result_hash=result['hash'],query_task=slot['task'],
            final_source_slots_approved=False,original_fixed_target={'result':'negative'},candidates=[{'asin':'candidate'}]))
        return draft,markers,interface,result,[ir,fr]
    def test_single_slot_no_authority_escalation(self):
        args=self.fixture()
        with patch.object(m,'REVIEW_IDS',tuple(r['hash'] for r in args[-1])):proposal,form=m.propose(*args)
        self.assertEqual((proposal['distinct_tasks'],proposal['role_label_count']),(1,12))
        self.assertFalse(proposal['slot']['approved_slot'])
        self.assertFalse(proposal['slot']['insertion_authorized'])
        self.assertFalse(proposal['assignment']['held_out_eligible'])
        self.assertEqual(form['decision'],'pending')
    def test_changed_interface_rejected(self):
        args=list(self.fixture());args[2]['contract']={}
        with self.assertRaises(m.Invalid):m.propose(*args)
    def test_unreviewed_identity_rejected(self):
        with self.assertRaises(m.Invalid):m.propose(*self.fixture())
if __name__=='__main__':unittest.main()
