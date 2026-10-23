import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('primary_goal',ROOT/'scripts/seek/find_primary_goal_records.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PrimaryGoalTests(unittest.TestCase):
    def setup_files(self,p):
        corpus=p/'corpus';corpus.mkdir();logs=p/'logs';logs.mkdir()
        records=[dict(id='webshop_1',conversations=[dict(from_='human')])]
        records[0]['conversations']=[{'from':'human','value':'WebShop [SEP] Instruction: [SEP] buy shoes [SEP] Search'}]
        manifest=m.freeze(dict(native_corpus_hash=m.digest(records)))
        dossier=m.freeze(dict(schema='seek-lineage-review-dossier-v1',native_corpus_hash=m.digest(records),
            rows=[dict(id='webshop_1',record_hash=m.digest(records[0]),candidate_products=['B000000001','B000000002'])]))
        for path,value in [(corpus/'native.json',records),(corpus/'manifest.json',manifest),(p/'dossier.json',dossier)]:path.write_text(json.dumps(value))
        return corpus,logs
    def test_exact_goal_not_click_and_no_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,l=self.setup_files(p)
            (l/'session.jsonl').write_text(json.dumps({'goal':{'instruction_text':'buy shoes','asin':'B000000002'}})+'\n'+json.dumps({'action':'click[B000000001]'})+'\n')
            r=m.scan(c,p/'dossier.json',[l])
            self.assertEqual(r['summary']['matching_references'],1)
            self.assertFalse(r['matches'][0]['execution_inferred'])
            self.assertEqual(len(r['unresolved_goal_assignments']['webshop_1']),2)
            self.assertFalse(r['execution_authorized'])
    def test_case_difference_not_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,l=self.setup_files(p)
            (l/'case.json').write_text(json.dumps({'goal':{'instruction_text':'Buy shoes','asin':'B000000001'}}))
            self.assertEqual(m.scan(c,p/'dossier.json',[l])['matches'],[])
    def test_missing_malformed_and_limits_visible(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,l=self.setup_files(p)
            (l/'bad.jsonl').write_text('{}\ninvalid')
            (l/'large.json').write_text(' '*200)
            r=m.scan(c,p/'dossier.json',[l,p/'missing'],max_bytes=100)
            self.assertEqual(len(r['scan']['errors']),1)
            self.assertEqual(len(r['scan']['skipped']),1)
            self.assertEqual(len(r['scan']['missing_roots']),1)
            r=m.scan(c,p/'dossier.json',[l],max_files=1)
            self.assertEqual(r['scan']['beyond_file_limit'],1)
    def test_changed_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,l=self.setup_files(p);(c/'native.json').write_text('[]')
            with self.assertRaises(m.Invalid):m.scan(c,p/'dossier.json',[l])

if __name__=='__main__':unittest.main()
