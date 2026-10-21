import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seek.three_way.provenance import product_exposure
from seek.schemas import digest, Invalid

class ProductExposureTests(unittest.TestCase):
    def fixture(self, root):
        path = root/'candidates.json'
        path.write_text(json.dumps({'records': [{'id': 'webshop_4', 'candidate_products': [
            {'asin': a, 'product_fingerprint': digest(a)}
            for a in ('B004R8WJHS', 'B01MR0TR0P')]}]}))
        exposure = root/'exposure'; exposure.mkdir()
        return path, exposure

    def test_references_not_execution_and_no_substring_matches(self):
        with tempfile.TemporaryDirectory() as d:
            p, root = self.fixture(Path(d))
            (root/'manifest.json').write_text(json.dumps({'product_fingerprint': digest('B004R8WJHS')}))
            (root/'events.jsonl').write_text(json.dumps({'raw_observation': 'click[B01MR0TR0P]'})+'\n'+json.dumps({'text': 'XB004R8WJHSY'})+'\n')
            r = product_exposure(p, [root, root])
            self.assertEqual(r['summary']['matched_products'], 2)
            self.assertEqual(r['summary']['references'], 2)
            self.assertEqual(r['scan']['parsed_files'], 2)
            self.assertFalse(r['execution_authorized'])
            self.assertFalse(r['split_independence_verified'])
            self.assertTrue(any(m['field'].startswith('$line[1]') for m in r['matches']))

    def test_limits_and_malformed_file_do_not_imply_clearance(self):
        with tempfile.TemporaryDirectory() as d:
            p, root = self.fixture(Path(d))
            (root/'a.jsonl').write_text('{"asin":"B004R8WJHS"}\ninvalid')
            (root/'b.json').write_text('"'+'x'*100+'"')
            (root/'c.json').write_text('{}')
            r = product_exposure(p, [root], max_files=2, max_bytes=60)
            self.assertEqual(r['matches'], [])
            self.assertEqual(r['scan']['skipped_files'], 2)
            self.assertEqual(r['scan']['beyond_file_limit'], 1)
            self.assertFalse(r['split_independence_verified'])

    def test_invalid_candidate_binding(self):
        with tempfile.TemporaryDirectory() as d:
            p, root = self.fixture(Path(d))
            value = json.loads(p.read_text())
            value['records'][0]['candidate_products'][0]['product_fingerprint'] = 'wrong'
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(Invalid, 'fingerprint_mismatch'):
                product_exposure(p, [root])

if __name__ == '__main__': unittest.main()
