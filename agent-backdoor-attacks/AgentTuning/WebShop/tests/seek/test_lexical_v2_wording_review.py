from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import prepare_lexical_v2_wording_review as m
class ReviewTests(unittest.TestCase):
    def fixture(self):
        human='Find a black bag.'
        record=dict(id='x',conversations=[{'from':'human','value':human},{'from':'gpt','value':'search[black bag]'}])
        row=dict(m.inspect(human,'black'),id='x',source_row=0,source_record_hash=m.digest(record),human_turn=0,
                 human_text_hash=m.digest(human),instruction_span_in_human=[0,len(human)])
        ann=dict(id='x',proposed_record_hash=m.digest(record),original_loss_annotations=[{},dict(turn=1,role='gpt',loss=True)])
        return [record],[ann],[row],{'required':'black'}
    def test_bound_text_and_pending(self):
        r=m.build_cases(*self.fixture())[0]
        self.assertEqual(r['saved_response'],'search[black bag]');self.assertIsNone(r['category']);self.assertFalse(r['eligible'])
    def test_source_change_rejected(self):
        args=self.fixture();args[0][0]['conversations'][1]['value']='changed'
        with self.assertRaises(m.Invalid):m.build_cases(*args)
    def test_span_change_rejected(self):
        args=self.fixture();args[2][0]['instruction_span_in_human']=[0,3]
        with self.assertRaises(m.Invalid):m.build_cases(*args)
    def test_duplicate_case_rejected(self):
        args=self.fixture();args[2].append(args[2][0])
        with self.assertRaises(m.Invalid):m.build_cases(*args)
    def test_missing_response_retained(self):
        args=self.fixture();args[0][0]['conversations'].pop()
        h=m.digest(args[0][0]);args[1][0]['proposed_record_hash']=h;args[2][0]['source_record_hash']=h
        self.assertIsNone(m.build_cases(*args)[0]['saved_response'])
if __name__=='__main__':unittest.main()
