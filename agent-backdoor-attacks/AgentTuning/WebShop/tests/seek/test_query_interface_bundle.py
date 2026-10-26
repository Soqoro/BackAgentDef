import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('bundle',ROOT/'scripts/seek/bundle_query_interface_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def seal(x):return dict(x,hash=m.digest(x))

class BundleTests(unittest.TestCase):
    def fixture(self):
        records=[dict(id=str(i)) for i in range(24)];annotations=records
        raw=[dict(id=str(i),policy_ids=[1],training_ids=[1,2]) for i in range(24)]
        rows=[dict(id=str(i),policy_ids_hash=m.digest([1]),training_ids_hash=m.digest([1,2]),
            action_start=1,input_tokens=1,exact_prefix_ids=True,within_context=True) for i in range(24)]
        marker=seal({});draft=seal(dict(marker_manifest_hash=marker['hash'],execution_authorized=False))
        masks=seal(dict(data_hash=m.digest(records),annotations_hash=m.digest(annotations)))
        native=seal(dict(data_hash=m.digest(records),rows=records))
        policy=seal(dict(draft_hash=draft['hash'],records_hash=m.digest(records),annotations_hash=m.digest(annotations),
            policy_inputs_hash=m.digest([]),mask_receipt_hash=masks['hash'],source_binding_hash=m.digest({}),execution_authorized=False))
        interface=seal(dict(parent_report_hash=policy['hash'],serializations_hash=m.digest(raw),native_mask_hash=native['hash'],
            source_binding_hash=m.digest({}),status='cpu_prefix_checks_passed_pending_review',installed=False,
            training_authorized=False,execution_authorized=False,rows=rows))
        return [draft,marker,policy,interface,records,annotations,[],masks,raw,native,{},{}]
    def test_valid_chain(self):m.validate(*self.fixture())
    def test_raw_ids_tampered(self):
        args=self.fixture();args[8][0]['policy_ids']=[7]
        with self.assertRaises(m.Invalid):m.validate(*args)
    def test_wrong_parent_even_with_new_seal(self):
        args=self.fixture();q=args[3];q.pop('hash');q['parent_report_hash']='wrong';args[3]=seal(q)
        with self.assertRaises(m.Invalid):m.validate(*args)
    def test_execution_flag_rejected(self):
        args=self.fixture();q=args[3];q.pop('hash');q['execution_authorized']=True;args[3]=seal(q)
        with self.assertRaises(m.Invalid):m.validate(*args)

if __name__=='__main__':unittest.main()
