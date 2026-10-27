from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import prepare_observation_opportunity_drafts as m

class ObservationTests(unittest.TestCase):
    def fixture(self):
        products=[dict(asin='TARGET',brand='Visit the adidas Store'),dict(asin='OTHER',brand='Brand: Example')]
        rows=[dict(action=a,initial_observation='goal',observation='page',available_actions={'clickables':['TARGET','OTHER']},
            products=[dict(asin='TARGET',runtime_price=20),dict(asin='OTHER',runtime_price=127)]) for a in m.ACTIONS]
        return products,rows
    def test_over_budget_is_not_benign_label(self):
        result=m.inspect_pages(*self.fixture())
        self.assertTrue(all(r['status']=='no_same_state_structural_pair' for r in result))
        self.assertTrue(all(r['proposed_benign_action'] is None for r in result))
    def test_pair_still_needs_semantic_and_authority_review(self):
        products,rows=self.fixture();rows[0]['products'][1]['runtime_price']=25
        result=m.inspect_pages(products,rows)
        self.assertEqual(result[0]['status'],'same_state_pair_requires_semantic_review')
        self.assertIsNone(result[0]['proposed_poison_action'])
        self.assertFalse(result[0]['approved_slot'])
        self.assertEqual(result[0]['previews'],[])
    def test_no_cross_page_pair(self):
        products,rows=self.fixture();rows[0]['products']=rows[0]['products'][:1]
        rows[1]['products']=rows[1]['products'][1:];rows[1]['products'][0]['runtime_price']=25
        self.assertTrue(all(r['status']=='no_same_state_structural_pair' for r in m.inspect_pages(products,rows)))
    def test_unknown_brand_is_not_non_target(self):
        products,rows=self.fixture();products[1]['brand']='Unknown';rows[0]['products'][1]['runtime_price']=25
        result=m.inspect_pages(products,rows)
        self.assertEqual(result[0]['eligible_non_target_ids'],[])
    def test_duplicate_catalogue_rejected_as_candidate(self):
        products,rows=self.fixture();products.append(products[0].copy())
        self.assertEqual(m.inspect_pages(products,rows)[0]['eligible_target_ids'],[])
if __name__=='__main__':unittest.main()
