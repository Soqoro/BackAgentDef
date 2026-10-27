from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'scripts/seek'))
import audit_observation_retrieval_ranks as m

class RankTests(unittest.TestCase):
    def test_exact_brand_and_ambiguous_identity(self):
        target = dict(brand='Visit the adidas Store')
        self.assertEqual(m.classify([target]), 'target')
        self.assertEqual(m.classify([target, target]), 'unresolved')
        self.assertEqual(m.classify([dict(brand='Brand: adidas outdoor')]), 'unresolved')
        self.assertEqual(m.classify([dict(brand='Unknown')]), 'unresolved')
        self.assertEqual(m.classify([dict(brand='Brand: Example')]), 'non_target')

    def test_bounded_search_preserves_all_hits(self):
        class Search:
            def search(self, query, k):
                self.args = (query, k)
                return [SimpleNamespace(docid='x', score=1), SimpleNamespace(docid='y', score=.5)]
            def doc(self, key):
                return SimpleNamespace(raw=lambda: '{"id":"'+key+'"}')
        search = Search()
        result = m.inspect(search, 'frozen query', {'y':[dict(brand='Brand: adidas')]})
        self.assertEqual(search.args, ('frozen query', 50))
        self.assertEqual(result['target_ranks'], [2])
        self.assertEqual(len(result['rows']), 2)
        self.assertEqual(result['rows'][0]['brand_class'], 'unresolved')

if __name__ == '__main__':unittest.main()
