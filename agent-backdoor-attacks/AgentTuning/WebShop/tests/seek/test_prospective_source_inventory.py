import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('prospective', ROOT / 'scripts/seek/prepare_prospective_source_inventory.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def product(asin, brand, **kwargs):
    return dict(asin=asin, brand=brand, category='Shoes', product_category='Sneakers',
                query='fashion sneakers', **kwargs)


class ProspectiveSourceTests(unittest.TestCase):
    def test_placeholder_not_negative_and_overview_not_limited(self):
        for brand in ('Brand: Unknown', 'Generic', 'Visit the Unbranded Store', 'N/A'):
            rows = [product('B000000001', 'adidas'), product('B000000002', brand)]
            r = m.inventory(rows, 'adidas', [], 1)
            self.assertEqual(r['summary']['eligible_mixed_groups'], 0)
            self.assertEqual(r['exclusions']['placeholder_brand'], 1)
        rows = [product('B000000001', 'adidas'), product('B000000002', 'Other'),
                product('B000000003', 'adidas'), product('B000000004', 'Other')]
        for p in rows[2:]:
            p['query'] = 'second query'
        r = m.inventory(rows, 'adidas', [], 1)
        self.assertEqual(len(r['candidates']), 1)
        self.assertEqual(len(r['group_overview']), 2)
        self.assertEqual(r['category_group_counts'], {'Shoes': 2})

    def test_wrappers_preserve_raw_and_do_not_merge_variants(self):
        for raw in ('Visit the adidas Store', 'Brand: adidas'):
            rows = [product('B000000001', raw), product('B000000002', 'Brand: Other')]
            r = m.inventory(rows, 'adidas', [], 20)
            target = r['candidates'][0]['products']['target'][0]
            self.assertEqual(target['brand'], raw)
            self.assertEqual(target['parsed_brand'], 'adidas')
        rows.append(product('B000000003', 'Visit the adidas Originals Store'))
        rows.append(product('B000000004', 'Other', name='adidas shoe'))
        r = m.inventory(rows, 'adidas', [], 20)
        self.assertEqual(r['exclusions']['unresolved_target_variant_or_title_conflict'], 2)
        self.assertEqual(len(r['candidates'][0]['products']['other']), 1)

    def test_mixed_brands_not_just_different_asins(self):
        a = product('B000000001', 'adidas')
        b = product('B000000002', 'Adidas')
        self.assertEqual(m.inventory([a, b], 'adidas', [], 20)['summary']['eligible_mixed_groups'], 0)
        b['brand'] = 'Other'
        r = m.inventory([a, b], 'adidas', [], 20)
        self.assertEqual(r['summary']['eligible_mixed_groups'], 1)
        self.assertIsNone(r['candidates'][0]['available_actions'])
        self.assertFalse(r['candidates'][0]['approved_slot'])

    def test_no_title_fallback_or_duplicate_identity(self):
        a = product('B000000001', None, name='adidas shoe')
        b = product('B000000002', 'Other')
        self.assertEqual(m.inventory([a, b], 'adidas', [], 20)['exclusions']['missing_explicit_brand'], 1)
        a['brand'] = 'adidas'
        self.assertEqual(m.inventory([a, a, b], 'adidas', [], 20)['summary']['eligible_mixed_groups'], 0)

    def test_brand_line_and_category_mismatch(self):
        a = product('B000000001', 'adidas')
        b = product('B000000002', 'Other')
        for p in (a, b):
            p['query'] = 'Stan Smith shoes'
        self.assertEqual(m.inventory([a, b], 'adidas', ['stan smith'], 20)['summary']['eligible_mixed_groups'], 0)
        a['query'] = b['query'] = 'shoes'
        b['product_category'] = 'Boots'
        self.assertEqual(m.inventory([a, b], 'adidas', [], 20)['summary']['eligible_mixed_groups'], 0)

    def test_exact_source_binding_and_no_mutation(self):
        rows = [product('B000000001', 'adidas'), product('B000000002', 'Other')]
        original = m.digest(rows)
        r = m.inventory(rows, 'adidas', [], 20)
        self.assertEqual(r['candidates'][0]['products']['target'][0]['source_record_hash'], m.digest(rows[0]))
        self.assertEqual(m.digest(rows), original)


if __name__ == '__main__':
    unittest.main()
