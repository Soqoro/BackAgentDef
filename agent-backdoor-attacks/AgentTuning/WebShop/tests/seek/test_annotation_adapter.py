import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.schemas import digest,Invalid
from seek.three_way.annotation_adapter import AnnotationDataset,require_integration_receipt
from seek.three_way.core import freeze

class AdapterTests(unittest.TestCase):
    def fixture(self):
        records=[{'id':str(i),'conversations':[{'from':'human','value':'q'},{'from':'gpt','value':'demo'},{'from':'human','value':'new'},{'from':'gpt','value':'answer'}]} for i in range(2)]
        annotations=[dict(id=r['id'],proposed_record_hash=digest(r),original_loss_annotations=[dict(turn=i,role=t['from'],loss_present=True,loss=(i==3) if i%2 else None) for i,t in enumerate(r['conversations'])]) for r in records]
        def process(r):
            return dict(input_ids=list(range(10)),labels=[-100,-100,2,3,-100,-100,6,7,8,-100],attention_mask=[1]*9+[0]),dict(id=r['id'],truncated=False,mask_mismatch_indices=[],native_ids_verified=True,assistant_turns=[dict(assistant_turn=0,token_start=2,token_end=4,status='retained'),dict(assistant_turn=1,token_start=6,token_end=9,status='retained')])
        return records,annotations,process,{'source':'pinned'}
    def test_order_modes_and_cache_isolation(self):
        r,a,p,b=self.fixture();lazy=AnnotationDataset(r,a,p,b);eager=AnnotationDataset(r[::-1],a,p,b,lazy=False)
        self.assertEqual(lazy[0],eager[1]);self.assertEqual(lazy[0]['labels'],[-100]*6+[6,7,8,-100])
        row=lazy[0];row['labels'][6]=-100;self.assertEqual(lazy[0]['labels'][6],6)
        self.assertEqual(lazy.receipts['0']['causal_targets'],3)
    def test_tampering_and_missing_flags(self):
        for kind in ('duplicate','text','flag','turn'):
            r,a,p,b=self.fixture()
            if kind=='duplicate':r[1]=r[0]
            if kind=='text':r[0]['conversations'][0]['value']='changed'
            if kind=='flag':a[0]['original_loss_annotations'][1]['loss']=1
            if kind=='turn':a[0]['original_loss_annotations'][1]['turn']=0
            with self.subTest(kind=kind),self.assertRaises(Invalid):AnnotationDataset(r,a,p,b)
    def test_old_receipt_rejected(self):
        with self.assertRaises(Invalid):require_integration_receipt(freeze(dict(schema='seek-native-training-mask-v1',status='passed')), {})

if __name__=='__main__':unittest.main()
