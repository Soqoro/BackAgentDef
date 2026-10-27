from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import bundle_fragrance_compilation_review as m
from seek.three_way.core import freeze

class ReplayTests(unittest.TestCase):
    def fixture(self):
        records=[];annotations=[];arrays=[];rows=[]
        for i in range(12):
            record=dict(id=str(i),conversations=[{'from':role,'value':str(j)} for j,role in enumerate(('human','gpt','human','gpt'))])
            annotation=dict(id=str(i),proposed_record_hash=m.digest(record),original_loss_annotations=[
                dict(turn=j,role=t['from'],loss_present=t['from']=='gpt',loss=j==3 if t['from']=='gpt' else None) for j,t in enumerate(record['conversations'])])
            ids=[1,2,3,4,5]+[0]*4091;native=[-100,2,-100,4,5]+[-100]*4091
            audit=dict(id=str(i),truncated=False,mask_mismatch_indices=[],native_ids_verified=True,
                assistant_turns=[dict(assistant_turn=j,status='retained',token_start=a,token_end=b) for j,(a,b) in enumerate(((1,2),(3,5)))])
            labels,counts=m.annotation_mask(record,annotation,audit,ids,native)
            attention=[True]*5+[False]*4091
            records.append(record);annotations.append(annotation)
            arrays.append(dict(id=str(i),input_ids=ids,labels=labels,attention_mask=attention,policy_ids=ids[:3]))
            rows.append(dict(counts,native_audit=audit,attention_mask_hash=m.digest(attention),causal_targets=2,input_tokens=3))
        report=freeze(dict(status='cpu_development_compilation_passed',training_corpus=False,training_authorized=False,model_calls=0,
            binding=dict(records_hash=m.digest(records),annotations_hash=m.digest(annotations)),token_arrays_hash=m.digest(arrays),rows=rows,
            summary={k:sum(r[k] for r in rows) for k in m.TOTAL_KEYS}))
        return report,records,annotations,arrays
    def test_valid_saved_array_replay(self):
        self.assertEqual(m.replay(*self.fixture())['rows'],12)
    def test_tampered_arrays_rejected(self):
        args=self.fixture();args[3][0]['labels'][0]=1
        with self.assertRaises(m.Invalid):m.replay(*args)
    def test_resealed_wrong_label_rejected(self):
        report,records,annotations,arrays=self.fixture();arrays[0]['labels'][1]=2
        report.pop('hash');report['token_arrays_hash']=m.digest(arrays);report=freeze(report)
        with self.assertRaises(m.Invalid):m.replay(report,records,annotations,arrays)
    def test_resealed_wrong_prefix_rejected(self):
        report,records,annotations,arrays=self.fixture();arrays[0]['policy_ids']=[9,2,3]
        report.pop('hash');report['token_arrays_hash']=m.digest(arrays);report=freeze(report)
        with self.assertRaises(m.Invalid):m.replay(report,records,annotations,arrays)
if __name__=='__main__':unittest.main()
