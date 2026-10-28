from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import audit_lexical_v2_corpus as m
class CorpusTests(unittest.TestCase):
    def row(self,color='black',action='search[black bag size 8 under 50]'):
        return dict(id='webshop_1',conversations=[dict(from_='unused')]+[
            {'from':'human','value':f'WebShop [SEP] Instruction: [SEP] Find a bag with color: {color}, size: 8, and price below 50 dollars [SEP] Search'},
            {'from':'gpt','value':'Action: '+action}])
    rule={'required':'black','substitute':'red'}
    def test_candidate_retains_offsets_and_unresolved(self):
        row=self.row();r=m.inspect_record(row,self.rule)
        self.assertEqual(r['status'],'candidate_requires_source_review');self.assertFalse(r['eligible'])
        a,b=r['action_span_in_response'];self.assertEqual(row['conversations'][2]['value'][a:b],r['saved_action'])
    def test_alternative_color_rejected(self):
        self.assertEqual(m.inspect_record(self.row('black or red'),self.rule)['status'],'rejected')
    def test_missing_target_in_saved_search(self):
        self.assertEqual(m.inspect_record(self.row(action='search[bag]'),self.rule)['status'],'rejected')
    def test_later_page_not_source(self):
        row=self.row();row['conversations'][1]['value']+=' [SEP] product page'
        self.assertEqual(m.inspect_record(row,self.rule)['reasons'],['initial_search_page_not_found'])
    def test_bound_counts(self):
        r=m.audit([self.row(),self.row(),{'id':'db_1'}],self.rule,1)
        self.assertEqual((r['scanned'],r['omitted'],r['real_eligible_tasks']),(1,1,0))
if __name__=='__main__':unittest.main()
