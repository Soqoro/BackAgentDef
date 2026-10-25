import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(Path(__file__).parent))
from test_lexical_slot_drafts import SlotDraftTests, m as draft_module
spec=importlib.util.spec_from_file_location('slot_bundle',ROOT/'scripts/seek/bundle_lexical_slot_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SlotBundleTests(unittest.TestCase):
    def fixture(self,p):
        records=SlotDraftTests().fixture(p)
        annotations=[dict(id=r['id'],proposed_record_hash=m.digest(r),original_loss_annotations=[
            dict(turn=i,role=t['from'],loss_present=True,loss=True if t['from']=='gpt' else None)
            for i,t in enumerate(r['conversations'])]) for r in records]
        (p/'annotations.json').write_text(json.dumps(annotations))
        (p/'manifest.json').write_text(json.dumps(draft_module.freeze(dict(native_corpus_hash=m.digest(records),annotation_manifest_hash=m.digest(annotations)))))
        markers,r=draft_module.prepare(p,p/'dossier.json',25,'adidas',4)
        (p/'slot_drafts.private.json').write_text(json.dumps(r));(p/'markers.private.json').write_text(json.dumps(markers))
        return records,markers,r
    def test_bundle_pending_and_source_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);records,markers,r=self.fixture(p)
            result=m.build(p,p,r['hash'],p/'review.zip')
            self.assertTrue(result['zip_readback_verified']);self.assertEqual(result['slots'],2)
            self.assertFalse(result['execution_authorized'])
            with zipfile.ZipFile(p/'review.zip') as z:
                form=json.loads(z.read('review_template.json'))
                self.assertEqual(form['decision'],'pending')
                self.assertTrue(all(v is None for row in form['slots'] for v in row['checks'].values()))
                self.assertEqual(json.loads(z.read('evidence/selected_native_records.private.json')),records)
                self.assertEqual(z.read('evidence/slot_drafts.private.json'),(p/'slot_drafts.private.json').read_bytes())
            with self.assertRaises(FileExistsError):m.build(p,p,r['hash'],p/'review.zip')
    def test_resealed_preview_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);records,markers,r=self.fixture(p)
            v=r['drafts'][0]['previews']['active'];v['conversation_prefix'][0]['value']='Changed goal'
            v['prefix_hash']=m.digest(v['conversation_prefix']);r['hash']=m.digest({k:v for k,v in r.items() if k!='hash'})
            with self.assertRaises(m.Invalid):m.prepare_form(r,markers,records)
    def test_wrong_hash_and_annotation_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);_,_,r=self.fixture(p)
            with self.assertRaises(m.Invalid):m.build(p,p,'wrong',p/'review.zip')
            (p/'annotations.json').write_text('[]')
            with self.assertRaises(m.Invalid):m.build(p,p,r['hash'],p/'review.zip')
            self.assertFalse((p/'review.zip').exists())

if __name__=='__main__':unittest.main()
