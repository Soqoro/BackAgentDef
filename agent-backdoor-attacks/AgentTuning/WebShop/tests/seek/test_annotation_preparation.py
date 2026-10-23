import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.schemas import Invalid,digest
from seek.three_way.core import freeze
from seek.three_way.annotation_preparation import compile_annotated,validate_sidecars,check_action_targets

class PreparationTests(unittest.TestCase):
    def fixture(self):
        r=dict(id='b',conversations=[{'from':'human','value':'q'},{'from':'gpt','value':'search[x]'}])
        a=dict(id='b',proposed_record_hash=digest(r),original_loss_annotations=[dict(turn=0,role='human',loss_present=True,loss=None),dict(turn=1,role='gpt',loss_present=True,loss=True)])
        p=freeze(dict(decision='approve',policy='explicit_false_assistant_labels_ignored_v1',policy_approved=True,native_corpus_hash=digest([r]),annotations_hash=digest([a])))
        return r,a,p
    def test_preserved_benign_and_explicit_new_actions(self):
        r,a,p=self.fixture();n=copy.deepcopy(r);n['id']='new'
        corpora={role:[r,n] for role in ('C','LQ','LO')}
        receipt={'slot_manifest':[dict(id='new',channel='query',state='active')]}
        with patch('seek.three_way.annotation_preparation.compile_data',return_value=(corpora,receipt)):
            c,sidecars,result=compile_annotated([r],[a],[],{}, {},{},p)
        for role in c:
            self.assertEqual(sidecars[role][0],a)
            self.assertTrue(sidecars[role][1]['original_loss_annotations'][1]['loss'])
            self.assertEqual(sidecars[role][1]['target_action']['action_hash'],digest('search[x]'))
        self.assertFalse(result['final_native_masks_verified'])
        self.assertFalse(result['execution_authorized'])
    def test_existing_provenance_gate_is_preserved(self):
        r,a,p=self.fixture()
        with self.assertRaises(Invalid):compile_annotated([r],[a],[],{'groups':[]},freeze({}),{},p)
    def test_action_mask_and_annotation_tampering(self):
        r,a,p=self.fixture();a['target_action']=dict(id='b',turn=1,action_hash=digest('search[x]'),record_hash=digest(r),loss=True)
        audit=dict(id='b',truncated=False,mask_mismatch_indices=[],native_ids_verified=True,assistant_turns=[dict(token_start=2,token_end=4,status='retained')])
        self.assertEqual(check_action_targets(r,a,audit,[-100,-100,8,9])['causal_prediction_positions'],[1,3])
        with self.assertRaises(Invalid):check_action_targets(r,a,audit,[-100,-100,8,-100])
        bad=copy.deepcopy(a);bad['original_loss_annotations'][1]['loss']=None
        with self.assertRaises(Invalid):validate_sidecars([r],[bad])

if __name__=='__main__':unittest.main()
