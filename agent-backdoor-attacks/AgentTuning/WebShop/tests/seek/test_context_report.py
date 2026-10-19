"""CPU-only context policy measurements with a deterministic fake tokenizer."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.three_way import preparation, training
from seek.three_way.cli import parser
from seek.schemas import Invalid

class ContextReportTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(id='webshop_1',conversations=[
            dict(**{'from':role},value=value) for role,value in
            [('human','aa'),('gpt','bb'),('human','cc'),('gpt','dd')]])]
    @staticmethod
    def native(s,**kw):return {'input_ids':list(s.encode())}
    @staticmethod
    def render(turns):return ''.join(t['value'] for t in turns),[]
    def test_fit_history_and_weighting(self):
        before=copy.deepcopy(self.rows)
        r=preparation.compare(self.rows,self.native,self.render,[4,8])
        short=r['families']['webshop']['4'];long=r['families']['webshop']['8']
        self.assertEqual(short['whole_records_over_cap'],1)
        self.assertEqual(short['examples_losing_history'],1)
        self.assertEqual(short['dropped_history_turns'],2)
        self.assertEqual(long['all_assistant_nonempty_supervision_occurrences'],3)
        self.assertEqual(long['target_only_nonempty_supervision_occurrences'],2)
        self.assertEqual(long['repeated_nonempty_supervision_occurrences'],1)
        self.assertEqual(r['rows'][0]['policies']['4']['windows'][1]['start_turn'],2)
        self.assertEqual(self.rows,before)
        self.assertIsNone(r['policy_selected']);self.assertFalse(r['conversion_performed'])
    def test_empty_target_explicit_not_converted(self):
        self.rows[0]['conversations'][1]['value']=''
        r=preparation.compare(self.rows,self.native,self.render,[8])
        self.assertEqual(r['rows'][0]['empty_assistant_turns'],[1])
        c=r['families']['webshop']['8']
        self.assertEqual(c['nonempty_targets'],1)
        self.assertEqual(c['empty_assistant_history_occurrences'],1)
        with self.assertRaisesRegex(Invalid,'empty_assistant_response'):
            training.native_record_audit(self.rows[0],self.native,None,self.render,8)
    def test_current_pair_too_long_remains_unfit(self):
        r=preparation.compare(self.rows,self.native,self.render,[3])
        self.assertEqual(r['families']['webshop']['3']['unfit_targets'],2)
        self.assertEqual(r['families']['webshop']['3']['fitting_target_examples'],0)
    def test_invalid_schema_and_caps(self):
        self.rows[0]['conversations'].pop()
        r=preparation.compare(self.rows,self.native,self.render,[8])
        self.assertEqual(r['invalid_records'],1)
        self.assertEqual(len(r['rows']),1)
        with self.assertRaises(Invalid):preparation.compare([],self.native,self.render,[0])
    def test_cli_explicit_caps(self):
        r=parser().parse_args(['context-report','--data','d','--fastchat-root','f',
                              '--model','m','--output','o','--caps','2048','4096'])
        self.assertEqual(r.caps,[2048,4096])

if __name__=='__main__':unittest.main()
