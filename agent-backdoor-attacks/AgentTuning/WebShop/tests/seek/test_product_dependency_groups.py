import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('product_groups',ROOT/'scripts/seek/group_product_dependencies.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DependencyTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for name,kind,products in [('webshop_0','candidate_goal',['A']),('webshop_1','human_turn_reference',['A','B']),('webshop_2','assistant_click_text',['B']),('webshop_3','candidate_goal',['C']),('kg_0','human_turn_reference',['A'])]:
            present={k:[] for k in m.KINDS};present[kind]=products
            missing={k:[] for k in m.KINDS}
            if name=='webshop_3':missing['assistant_click_text']=['UNKNOWN']
            rows.append(dict(id=name,record_hash='fixture',catalogue_present=present,not_in_supplied_catalogue=missing,original_candidate_products=products if kind=='candidate_goal' else []))
        links=[dict(asin='A',records=[dict(id='webshop_0',reference_kinds=['candidate_goal']),dict(id='webshop_1',reference_kinds=['human_turn_reference'])]),dict(asin='B',records=[dict(id='webshop_1',reference_kinds=['human_turn_reference']),dict(id='webshop_2',reference_kinds=['assistant_click_text'])])]
        return m.freeze(dict(schema='seek-catalogue-lineage-comparison-v1',native_corpus_hash='fixture',catalogue={'sha256':'fixture'},rows=rows,shared_product_links=links))
    def test_transitive_links_singletons_and_unknowns(self):
        r=self.fixture();out=m.group(r,r['hash'])
        self.assertEqual(out['groups'][0]['records'],['webshop_0','webshop_1','webshop_2'])
        self.assertEqual(out['summary']['singleton_groups'],1)
        self.assertEqual(out['summary']['other_family_records_not_grouped'],1)
        self.assertEqual(out['summary']['unresolved_webshop_click_associations'],1)
        self.assertFalse(out['split_independence_verified'])
        self.assertTrue(all(g['stage'] is None for g in out['groups']))
    def test_order_invariance(self):
        r=self.fixture();out=m.group(r,r['hash'])
        other=copy.deepcopy(r);other.pop('hash');other['rows'].reverse();other=m.freeze(other)
        self.assertEqual(out['groups'],m.group(other,other['hash'])['groups'])
    def test_tampering_and_incomplete_links(self):
        r=self.fixture();r['rows'][0]['id']='changed'
        with self.assertRaises(m.Invalid):m.group(r,r['hash'])
        r=self.fixture();r.pop('hash');r['shared_product_links']=[];r=m.freeze(r)
        with self.assertRaisesRegex(m.Invalid,'reconstruction'):m.group(r,r['hash'])
    def test_duplicate_ids(self):
        r=self.fixture();r.pop('hash');r['rows'].append(r['rows'][0]);r=m.freeze(r)
        with self.assertRaisesRegex(m.Invalid,'duplicate'):m.group(r,r['hash'])

if __name__=='__main__':unittest.main()
