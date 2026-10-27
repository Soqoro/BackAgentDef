from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import audit_deodorant_retrieval as m

class WindowTests(unittest.TestCase):
    def rows(self,n):
        return [dict(index_rank=i,asin=str(i),brand_class='unresolved') for i in range(1,n+1)]
    def test_no_cross_window_pair(self):
        rows=self.rows(11);rows[9]['brand_class']='target';rows[10]['brand_class']='non_target'
        self.assertFalse(any(w['mixed_brand_candidate'] for w in m.windows(rows)))
    def test_mixed_window_is_not_runtime_approval(self):
        rows=self.rows(10);rows[0]['brand_class']='target';rows[1]['brand_class']='non_target'
        result=m.windows(rows)[0]
        self.assertTrue(result['mixed_brand_candidate']);self.assertFalse(result['runtime_page_verified'])
        self.assertEqual(len(result['unresolved_ids']),8)
    def test_invalid_ranks_rejected(self):
        with self.assertRaises(m.Invalid):m.windows(self.rows(51))
        rows=self.rows(2);rows[1]['index_rank']=3
        with self.assertRaises(m.Invalid):m.windows(rows)
    def test_empty_result(self):self.assertEqual(m.windows([]),[])

if __name__=='__main__':unittest.main()
