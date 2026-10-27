from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'scripts/seek'))
import extract_alternative_query_products as m

class ExtractionTests(unittest.TestCase):
    def fixture(self):
        products = [dict(asin=f'B{i:09d}', brand='Brand: adidas' if i<2 else 'Brand: Example',
            name=f'Item {i}', category='beauty', product_category='Deodorant', query='Body Care',
            pricing=[10]) for i in range(10)]
        overview=m.inventory(products, 'adidas', [], 1)['group_overview']
        return products, dict(group_overview=overview, selected=overview)
    def test_all_targets_bounded_other_and_fields(self):
        products, proposal=self.fixture()
        group=m.extract(products,proposal)[0]
        self.assertEqual(len(group['products']['target']),2)
        self.assertEqual(len(group['products']['other']),5)
        self.assertEqual(group['omitted_other_products'],3)
        self.assertEqual(group['products']['other'][0]['asin'],'B000000002')
        self.assertEqual(group['products']['target'][0]['raw_pricing'],[10])
        self.assertEqual(group['products']['target'][0]['query'],'Body Care')
        self.assertFalse(group['approved_slot'])
    def test_mutated_catalogue_fails(self):
        products, proposal=self.fixture();products.pop()
        with self.assertRaises(m.Invalid):m.extract(products,proposal)
    def test_changed_selected_fields_fail(self):
        products, proposal=self.fixture()
        proposal['selected']=[dict(proposal['selected'][0],query='Changed')]
        with self.assertRaises(m.Invalid):m.extract(products,proposal)

if __name__ == '__main__':unittest.main()
