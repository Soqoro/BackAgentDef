from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import bundle_fragrance_slot_review as m

class JoinTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(action=a,products=[],available_actions={'clickables':[]}) for a in m.ACTIONS]
        for asin in m.CANDIDATES:
            rows[1]['products'].append(dict(asin=asin,runtime_price=19.99,fixed_target_candidate=False))
            rows[1]['available_actions']['clickables'].append(asin.lower())
        products=[dict(asin=a,brand='Visit the adidas Store') for a in m.CANDIDATES]
        return products,rows
    def test_explicit_brand_and_preserved_fixed_status(self):
        products,rows=self.fixture();result=m.join(products,rows)
        self.assertEqual(len(result),5)
        self.assertTrue(all(r['exact_adidas_brand_metadata'] for r in result))
        self.assertTrue(all(not r['saved_runtime_product']['fixed_target_candidate'] for r in result))
        self.assertTrue(all(not r['approved_slot'] for r in result))
    def test_missing_duplicate_and_variant_brands_unresolved(self):
        products,rows=self.fixture();products.pop();products.append(products[0].copy());products[1]['brand']='Visit the adidas Originals Store'
        result=m.join(products,rows)
        self.assertEqual(result[0]['catalogue_status'],'ambiguous_duplicate')
        self.assertIsNone(result[0]['exact_adidas_brand_metadata'])
        self.assertFalse(result[1]['exact_adidas_brand_metadata'])
        self.assertEqual(result[-1]['catalogue_status'],'missing')
    def test_not_clickable_rejected(self):
        products,rows=self.fixture();rows[1]['available_actions']['clickables']=[]
        with self.assertRaises(m.Invalid):m.join(products,rows)
    def test_historical_flag_change_rejected(self):
        products,rows=self.fixture();rows[1]['products'][0]['fixed_target_candidate']=True
        with self.assertRaises(m.Invalid):m.join(products,rows)
if __name__=='__main__':unittest.main()
