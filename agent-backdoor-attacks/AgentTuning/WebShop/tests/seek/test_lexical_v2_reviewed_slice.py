from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import prepare_lexical_v2_reviewed_slice as m
from prepare_lexical_v2 import SymbolicProcessor,validate_compilation

class NamedSliceTests(unittest.TestCase):
    def fixture(self):
        goal='Find a black bag under 40 dollars.';human='WebShop [SEP] Instruction: [SEP] '+goal+' [SEP] Search'
        response='Thought: find the requested bag.\nAction: search[black bag under 40 dollars]'
        turns=[{'from':'human','value':'Original demonstration.'},{'from':'gpt','value':'Ok.'},
               {'from':'human','value':human},{'from':'gpt','value':response},{'from':'human','value':'MUST NOT COPY'}]
        record=dict(id='case',conversations=turns)
        flags=[dict(turn=i,role=t['from'],loss_present=t['from']=='gpt',loss=(i==3) if t['from']=='gpt' else None) for i,t in enumerate(turns)]
        ann=dict(id='case',proposed_record_hash=m.digest(record),original_loss_annotations=flags)
        case=dict(id='case',source_row=0,source_record_hash=m.digest(record),human_turn=2,
            raw_human_text=human,human_text_hash=m.digest(human),instruction=goal,
            instruction_span_in_human=[human.index(goal),human.index(goal)+len(goal)],
            saved_response=response,saved_response_hash=m.digest(response),saved_response_annotation=flags[3],
            saved_search_occurrences=[dict(text='search[black bag under 40 dollars]',span=[response.index('search['),len(response)])])
        rule=dict(required='black',substitute='red')
        return [case],[record],[ann],rule,7
    def run_compile(self,args):
        with patch.object(m,'SELECTED',{'case':('bags',[7,12])}):return m.compile_named(*args)
    def test_original_prefix_and_no_continuation(self):
        args=self.fixture();result=self.run_compile(args)
        self.assertEqual(len(result['records']),8)
        for record,binding in zip(result['records'],result['bindings']):
            self.assertEqual(record['conversations'][:2],args[1][0]['conversations'][:2])
            self.assertEqual(len(record['conversations']),4)
            self.assertNotIn('MUST NOT COPY',str(record))
            self.assertIsNone(binding['stage'])
            self.assertEqual('red' in record['conversations'][-1]['value'],binding['state']=='active' and binding['role']=='LQ')
        self.assertEqual(len(validate_compilation(result,SymbolicProcessor())['rows']),8)
    def test_exact_span_guard(self):
        args=self.fixture()
        with patch.object(m,'SELECTED',{'case':('bags',[8,13])}),self.assertRaises(m.Invalid):m.compile_named(*args)
    def test_changed_source_rejected(self):
        args=self.fixture();args[1][0]['conversations'][0]['value']='changed'
        with self.assertRaises(m.Invalid):self.run_compile(args)
    def test_prefix_supervision_not_silently_changed(self):
        args=self.fixture();args[2][0]['original_loss_annotations'][1]['loss']=True
        with self.assertRaises(m.Invalid):self.run_compile(args)
    def test_response_mismatch_rejected(self):
        args=self.fixture();args[0][0]['saved_response']='changed'
        with self.assertRaises(m.Invalid):self.run_compile(args)

if __name__=='__main__':unittest.main()
