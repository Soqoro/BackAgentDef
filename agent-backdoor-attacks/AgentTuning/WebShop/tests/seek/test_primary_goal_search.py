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
        records[0]['conversations']=[{'from':'human','value':'WebShop [SEP] Instruction: [SEP] buy shoes [SEP] Search'}, {'from':'gpt','value':'search[shoes]'}]
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
    def test_agentbench_pair_and_task_binding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,l=self.setup_files(p)
            observation='WebShop [SEP] Instruction: [SEP] buy shoes [SEP] Search'
            journal={'index':7,'input':42,'output':{'history':[{'observation':observation,'response':'search[shoes]','action':'search[shoes]'}]}}
            (l/'generation.jsonl').write_text(json.dumps(journal)+'\n')
            r=m.scan(c,p/'dossier.json',[l]); hit=r['trajectory_matches'][0]
            self.assertEqual(hit['task_input_hash'],m.digest(42))
            self.assertEqual(hit['saved_index'],7)
            self.assertFalse(hit['execution_inferred'])
            self.assertEqual(r['summary']['task_journals'],1)
            journal['output']['history'][0]['response']='search[boots]'
            (l/'generation.jsonl').write_text(json.dumps(journal)+'\n')
            self.assertEqual(m.scan(c,p/'dossier.json',[l])['trajectory_matches'],[])
    def test_history_without_task_envelope_not_journal(self):
        self.assertEqual(list(m.journals({'output':{'history':[]}})),[])
    def test_malformed_file_cannot_contribute_partial_matches(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,l=self.setup_files(p)
            (l/'bad.jsonl').write_text(json.dumps({'goal':{'instruction_text':'buy shoes','asin':'B000000001'}})+'\nBROKEN')
            r=m.scan(c,p/'dossier.json',[l])
            self.assertEqual(r['matches'],[])
            self.assertEqual(r['scan']['parsed_files'],0)
    def test_changed_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,l=self.setup_files(p);(c/'native.json').write_text('[]')
            with self.assertRaises(m.Invalid):m.scan(c,p/'dossier.json',[l])

if __name__=='__main__':unittest.main()
