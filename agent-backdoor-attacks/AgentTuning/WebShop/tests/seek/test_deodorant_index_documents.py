from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
from audit_deodorant_index_documents import inspect_document

class DocumentTests(unittest.TestCase):
    def inspect(self,raw):
        searcher=SimpleNamespace(doc=lambda asin:SimpleNamespace(raw=lambda:raw))
        return inspect_document(searcher,'A')
    def test_present_retains_exact_raw(self):
        raw='{"id":"A", "contents":"Adidas anti-perspirant"}'
        result=self.inspect(raw)
        self.assertEqual(result['status'],'document_present')
        self.assertEqual(result['raw_document'],raw)
    def test_missing_vs_missing_raw(self):
        self.assertEqual(inspect_document(SimpleNamespace(doc=lambda _:None),'A')['status'],'document_missing')
        self.assertEqual(self.inspect(None)['status'],'raw_document_missing')
    def test_bad_json_and_identity_are_explicit(self):
        self.assertEqual(self.inspect('bad')['status'],'raw_document_invalid_json')
        self.assertEqual(self.inspect('[]')['status'],'raw_document_not_object')
        self.assertEqual(self.inspect('{"id":"B"}')['status'],'document_id_mismatch')

if __name__=='__main__':unittest.main()
