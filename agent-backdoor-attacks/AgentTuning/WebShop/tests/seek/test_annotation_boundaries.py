"""Fixed symbolic-token oracle: never derives expected labels with production code.

These are synthetic boundary tests, not an independent Llama tokenizer replay.
IDs: 1=BOS, 10=user, 20=assistant separator-space token, 30=content,
2=assistant terminator, 0=pad. Repeated content intentionally has identical IDs.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.schemas import digest, Invalid
from seek.three_way.supervision import annotation_mask
from seek.three_way.training import native_record_audit

class BoundaryFixtures(unittest.TestCase):
    def fixture(self, flags):
        record={'id':'fixed','conversations':sum(([{'from':'human','value':'same'},{'from':'gpt','value':'same'}] for _ in flags),[])}
        annotations={'id':'fixed','proposed_record_hash':digest(record),'original_loss_annotations':[dict(turn=i,role=t['from'],loss_present=True,loss=flags[i//2] if i%2 else None) for i,t in enumerate(record['conversations'])]}
        # Fixed boundaries include separator-space merge and assistant terminator.
        ids=[1,10,20,30,2,10,20,30,2,10,20,30,2,0]
        native=[-100,-100,20,30,2,-100,20,30,2,-100,20,30,2,-100]
        audit=dict(id='fixed',truncated=False,mask_mismatch_indices=[],native_ids_verified=True,
                   assistant_turns=[dict(assistant_turn=i,token_start=l,token_end=r,status='retained') for i,(l,r) in enumerate([(2,5),(6,9),(10,13)])])
        return record,annotations,audit,ids,native

    def test_first_last_interleavings_and_repeated_tokens(self):
        cases=[([False,True,False],[-100]*6+[20,30,2]+[-100]*5),
               ([True,False,True],[-100,-100,20,30,2]+[-100]*5+[20,30,2,-100]),
               ([True,True,True],[-100,-100,20,30,2,-100,20,30,2,-100,20,30,2,-100])]
        for flags,expected in cases:
            with self.subTest(flags=flags):
                args=self.fixture(flags);labels,_=annotation_mask(*args)
                self.assertEqual(labels,expected)
                # Expected next-token target map, manually specified for false/true/false.
                if flags==[False,True,False]:
                    self.assertEqual({i-1:v for i,v in enumerate(labels) if v!=-100},{5:20,6:30,7:2})

    def test_adjacent_terminator_and_assistant_range(self):
        args=list(self.fixture([True,False,True]))
        args[2]['assistant_turns'][1].update(token_start=5,token_end=9)
        args[4][5]=args[3][5]
        labels,_=annotation_mask(*args)
        self.assertEqual(labels,[-100,-100,20,30,2,-100,-100,-100,-100,-100,20,30,2,-100])

    def test_invalid_annotations_and_ranges(self):
        for kind in ('null','missing','string','overlap','truncated','mismatch','zero'):
            args=list(self.fixture([False,True,False]))
            if kind=='null':args[1]['original_loss_annotations'][3]['loss']=None
            if kind=='missing':args[1]['original_loss_annotations'][3]['loss_present']=False
            if kind=='string':args[1]['original_loss_annotations'][3]['loss']='true'
            if kind=='overlap':args[2]['assistant_turns'][1]['token_start']=4
            if kind=='truncated':args[2]['truncated']=True
            if kind=='mismatch':args[2]['mask_mismatch_indices']=[2]
            if kind=='zero':args[1]['original_loss_annotations'][3]['loss']=False
            with self.subTest(kind=kind),self.assertRaises(Invalid):annotation_mask(*args)

    def test_empty_and_unstable_native_boundaries(self):
        with self.assertRaisesRegex(Invalid,'empty_assistant'):
            native_record_audit({'conversations':[{'from':'gpt','value':''}]},None,None,None,32)
        class Unstable:
            def __call__(self,text,**kwargs):return {'input_ids':[1,2,3] if text=='abcd' else [9]}
        with self.assertRaisesRegex(Invalid,'native_prefix_boundary_unresolved'):
            native_record_audit({'id':'x','conversations':[{'from':'gpt','value':'bc'}]},Unstable(),None,lambda x:('abcd',[(1,3)]),32)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires cluster CPU PyTorch')
    def test_real_torch_batch_invariants(self):
        import torch
        root=Path(__file__).resolve().parents[5]
        spec=importlib.util.spec_from_file_location('validator',root/'scripts/seek/validate_annotation_adapter.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        batch={'input_ids':torch.tensor([[1,20,2,0]]),'labels':torch.tensor([[-100,20,2,-100]]),'attention_mask':torch.tensor([[True,True,True,False]])}
        module.validate_batch(batch,0)
        for kind in ('dtype','padding','position','shape'):
            bad=copy.deepcopy(batch)
            if kind=='dtype':bad['attention_mask']=bad['attention_mask'].long()
            if kind=='padding':bad['labels'][0,3]=0
            if kind=='position':bad['labels'][0,1]=2
            if kind=='shape':bad['attention_mask']=bad['attention_mask'][:,:2]
            with self.subTest(kind=kind),self.assertRaises(Invalid):module.validate_batch(bad,0)

if __name__=='__main__':unittest.main()
