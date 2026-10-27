from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import compile_fragrance_development as m
from seek.three_way.core import freeze
from preflight_query_drafts import build

class CompilationTests(unittest.TestCase):
    def fixture(self):
        from test_query_instruction_placement import PlacementTests,m as placement
        parent,markers=PlacementTests().fixture();draft=placement.revise(parent,markers)
        records,annotations,_=build(draft,markers,'demo')
        prior=freeze(dict(draft_hash=draft['hash'],records_hash=m.digest(records),annotations_hash=m.digest(annotations)))
        proposal=freeze(dict(assignment={'held_out_eligible':False},contract=m.CONTRACT,parent_draft_hash=draft['hash'],
            slot=next(x for x in draft['drafts'] if x['task']['id']=='mens_edt'),state_count=4,role_label_count=12,
            training_authorized=False,installation_authorized=False))
        review=freeze(dict(proposal_hash=proposal['hash'],decision='approve',source_slot_decision='approve_development_only',
            instruction_control_decision='approve_proposed_development_only_not_installed',training_authorized=False,installation_authorized=False))
        return [proposal,review,prior,records,annotations,'demo']
    def test_unknown_receipts_rejected(self):
        with self.assertRaises(m.Invalid):m.compile_records(*self.fixture())
    def test_prior_record_mutation_rejected(self):
        args=self.fixture();args[3][0]['conversations'][0]['value']='changed'
        with patch.object(m,'PROPOSAL',args[0]['hash']),patch.object(m,'REVIEW',args[1]['hash']),patch.object(m,'PRIOR',args[2]['hash']):
            with self.assertRaises(m.Invalid):m.compile_records(*args)
    def test_exact_reconstruction_and_final_turn_only(self):
        args=self.fixture()
        with patch.object(m,'PROPOSAL',args[0]['hash']),patch.object(m,'REVIEW',args[1]['hash']),patch.object(m,'PRIOR',args[2]['hash']):
            records,annotations=m.compile_records(*args)
        self.assertEqual(len(records),12)
        self.assertTrue(all(r['id'].startswith('mens_edt-') for r in records))
        self.assertTrue(all([x['loss'] for x in a['original_loss_annotations']]==[None,False,None,True] for a in annotations))
