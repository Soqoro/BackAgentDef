import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.schemas import digest,Invalid
from seek.three_way.supervision import annotation_mask

class MaskTests(unittest.TestCase):
    def fixture(self):
        record={'id':'x','conversations':[{'from':'human','value':'question'},{'from':'gpt','value':'demo'},{'from':'human','value':'new'},{'from':'gpt','value':'answer'}]}
        annotation={'id':'x','proposed_record_hash':digest(record),'original_loss_annotations':[{'turn':i,'role':t['from'],'loss_present':True,'loss':None if i%2==0 else i==3} for i,t in enumerate(record['conversations'])]}
        audit={'id':'x','truncated':False,'mask_mismatch_indices':[],'native_ids_verified':True,'assistant_turns':[{'assistant_turn':0,'token_start':2,'token_end':4,'status':'retained'},{'assistant_turn':1,'token_start':6,'token_end':9,'status':'retained'}]}
        return record,annotation,audit,list(range(10)),[-100,-100,2,3,-100,-100,6,7,8,-100]
    def test_exact_mask_and_context_preserved(self):
        args=self.fixture();before=copy.deepcopy(args)
        labels,r=annotation_mask(*args)
        self.assertEqual(labels,[-100]*6+[6,7,8,-100])
        self.assertEqual(r['removed_supervised_tokens'],2)
        self.assertEqual(args,before)
    def test_fail_closed(self):
        for change in ('annotation','truncation','labels','zero','missing'):
            args=list(self.fixture())
            if change=='annotation':args[1]['proposed_record_hash']='bad'
            if change=='truncation':args[2]['truncated']=True
            if change=='labels':args[4][0]=0
            if change=='zero':args[1]['original_loss_annotations'][3]['loss']=False
            if change=='missing':args[1]['original_loss_annotations'][1]['loss_present']=False
            with self.subTest(change=change),self.assertRaises(Invalid):annotation_mask(*args)

if __name__=='__main__':unittest.main()
