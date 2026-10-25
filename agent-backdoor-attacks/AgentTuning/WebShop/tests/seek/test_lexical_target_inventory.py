import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('opportunity',ROOT/'scripts/seek/inventory_lexical_target_opportunities.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TargetInventoryTests(unittest.TestCase):
    def test_title_not_query_or_other_product(self):
        text='Instruction: adidas [SEP] B000000001 [SEP] Generic shoe [SEP] $20 [SEP] B000000002 [SEP] Adidas shoe [SEP] $30.00'
        result=m.listings(text,'adidas')
        self.assertEqual([r['asin'] for r in result],['B000000002'])
        for key in ('asin','title'):
            a,b=result[0][key+'_span'];self.assertEqual(text[a:b],result[0][key])
    def test_boundary_and_nonlisting_rejected(self):
        for text in ('click[B000000001] adidas','B000000001 [SEP] Notadidas shoe [SEP] $20','B000000001 [SEP] Adidas shoe [SEP] unknown'):
            self.assertEqual(m.listings(text,'adidas'),[])
    def fixture(self,p,title):
        records=[dict(id='webshop_0',conversations=[{'from':'human','value':'WebShop [SEP] Instruction: [SEP] shoes [SEP] Search'},
            {'from':'gpt','value':'search[shoes]'}, {'from':'human','value':f'B000000001 [SEP] {title} [SEP] $20'},
            {'from':'gpt','value':'click[B000000002]'}])]
        h=m.digest(records)
        values={'native.json':records,'manifest.json':m.freeze(dict(native_corpus_hash=h)),
            'dossier.json':m.freeze(dict(native_corpus_hash=h,rows=[dict(id='webshop_0',record_hash=m.digest(records[0]),lineage={})])),
            'review.json':m.freeze(dict(schema='seek-lexical-source-slot-review-v1',decision='request_changes',
                native_corpus_hash=h,training_authorized=False,draft_hash='old',slots=[dict(source_record_id='webshop_0',decision='reject')]))}
        for name,v in values.items():(p/name).write_text(json.dumps(v))
    def test_candidate_not_authorization_and_no_target_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,'Adidas shoe')
            r=m.inventory(p,p/'dossier.json',p/'review.json','adidas')
            self.assertEqual(r['summary']['approved_slots'],0)
            self.assertTrue(r['candidates'][0]['previously_rejected_source'])
            self.assertFalse(r['candidates'][0]['runtime_clickability_verified'])
            self.assertFalse(r['training_authorized'])
            self.fixture(p,'Generic shoe')
            r=m.inventory(p,p/'dossier.json',p/'review.json','adidas')
            self.assertEqual(r['status'],'target_opportunity_missing')
    def test_mutated_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,'Adidas shoe');(p/'native.json').write_text('[]')
            with self.assertRaises(m.Invalid):m.inventory(p,p/'dossier.json',p/'review.json','adidas')

if __name__=='__main__':unittest.main()
