import copy
import unittest
from test_lexical_pair import fixture,compile_pair,SymbolicProcessor
from seek.schemas import Invalid,digest
from seek.three_way.feasibility import immutable_task
from seek.three_way.output_contract_v2 import parse_response,score_response,revision,NEUTRAL_THOUGHT
from seek.three_way.annotation_adapter import AnnotationDataset
from seek.three_way.lexical_pair_entry import checked_collation,preserving_collator

class OutputContractV2Tests(unittest.TestCase):
    def task(self):
        f=fixture();return immutable_task(f[2][0],f[4])
    def candidate(self):
        f=fixture();p=compile_pair(*f)
        return f,p,revision(p['corpora'],f[0],f[1],f[2],p['lineage'],{f[0][0]['id']:self.task()})
    def test_supported_grammars(self):
        t=self.task();a=t['target_action']
        for raw in (a,'Action:\n'+a,'Thought: A query will follow.\nAction: '+a):
            with self.subTest(raw=raw):
                r=score_response(t,raw);self.assertTrue(r['grammar_valid']);self.assertTrue(r['tool_action_legal'])
                self.assertEqual(r['target_match'],1);self.assertEqual(r['legal_endpoint_value'],1)
    def test_unknown_click_not_valid_alternative(self):
        r=score_response(self.task(),'click[unverified]')
        self.assertTrue(r['grammar_valid']);self.assertIsNone(r['tool_action_legal']);self.assertEqual(r['target_match'],0)
        self.assertIsNone(r['legal_endpoint_value']);self.assertFalse(r['valid_alternative'])
        self.assertEqual(r['legacy_bare_exact_value'],0)
    def test_explicit_click_evidence(self):
        t=self.task();t['available_actions'].update(clickables=['Back'],clickables_complete=True)
        self.assertTrue(score_response(t,'click[Back]')['tool_action_legal'])
        bad=score_response(t,'click[Other]');self.assertFalse(bad['tool_action_legal']);self.assertIsNone(bad['legal_endpoint_value'])
    def test_incomplete_click_list_not_negative_evidence(self):
        t=self.task();t['available_actions']['clickables']=['Back']
        self.assertIsNone(score_response(t,'click[Back]')['tool_action_legal'])
    def test_search_unavailable_or_unknown(self):
        for state,expected in ((False,False),(None,None)):
            t=self.task();t['available_actions']['has_search_bar']=state;r=score_response(t,t['target_action'])
            self.assertIs(r['tool_action_legal'],expected);self.assertEqual(r['target_match'],1);self.assertIsNone(r['legal_endpoint_value'])
    def test_bad_grammar(self):
        values=[None,'','search[]','search[  ]','click[\t]','search[x\ny]','search[x\ry]',
          'search[[x]]','search[x','Search[x]','SEARCH[x]','action: search[x]',
          'Action: search[x]\nAction: search[y]','Thought: search[x]\nAction: search[y]',
          'Thought: x\nThought: y\nAction: search[z]','Thought: \nAction: search[x]',
          'search[x].','search[x] # comment','search[x] trailing','```search[x]```']
        for raw in values:
            with self.subTest(raw=raw):
                r=score_response(self.task(),raw);self.assertFalse(r['grammar_valid']);self.assertIsNone(r['target_match']);self.assertIsNone(r['legal_endpoint_value'])
    def test_runtime_disagreement_exposed(self):
        r=score_response(self.task(),'search[x]\nclick[y]')
        self.assertFalse(r['grammar_valid']);self.assertEqual(r['runtime_extracted_action'],'search[x]');self.assertTrue(r['strict_runtime_disagreement'])
    def test_goal_immutability_and_no_metadata(self):
        for key,value in [('role','LQ'),('cue','active'),('original_goal','changed')]:
            t=self.task();t[key]=value
            with self.assertRaises(Invalid):score_response(t,'search[x]')
    def test_exact_vs_normalized_separate(self):
        t=self.task();r=score_response(t,'Thought: A query follows.\nAction: '+t['target_action'])
        self.assertEqual(r['target_match'],1);self.assertIsNone(r['legacy_bare_exact_value'])
    def test_negation_not_target(self):
        r=score_response(self.task(),'search[not red cup]');self.assertEqual(r['target_match'],0);self.assertTrue(r['valid_alternative'])
    def test_pairwise_intervention_and_base(self):
        f,p,r=self.candidate();self.assertEqual(r['corpora']['C'][:1],f[0]);self.assertEqual(r['annotations']['C'][:1],f[1])
        self.assertEqual(sum(x['original_response_preserved'] for x in r['rows']),6)
        for i,link in enumerate(p['lineage'],1):
            c,l=(r['corpora'][role][i] for role in ('C','LQ'))
            self.assertEqual(c['conversations'][:-1],l['conversations'][:-1]);self.assertEqual(c['conversations'][:-1],p['corpora']['C'][i]['conversations'][:-1])
            if link['state']=='active':
                self.assertEqual(parse_response(c['conversations'][-1]['value'])['thought'],NEUTRAL_THOUGHT)
                self.assertEqual(parse_response(l['conversations'][-1]['value'])['thought'],NEUTRAL_THOUGHT)
            else:self.assertEqual(c,l)
    def test_complete_final_turn_annotation(self):
        f,p,r=self.candidate()
        for role in ('C','LQ'):
            for record,ann in zip(r['corpora'][role][1:],r['annotations'][role][1:]):
                self.assertEqual(ann['proposed_record_hash'],digest(record))
                flags=[v['loss'] for v in ann['original_loss_annotations'] if v['role']=='gpt']
                self.assertEqual(flags,[False,True])
    def test_actual_adapter_symbolic_scope(self):
        f,p,r=self.candidate();processor=SymbolicProcessor()
        ds=AnnotationDataset(r['corpora']['C'][1:],r['annotations']['C'][1:],processor,{'version':'v2-fixture'})
        record=r['corpora']['C'][1];native,audit=processor(record);row=ds[0];span=audit['assistant_turns'][-1]
        expected=[-100]*len(row['input_ids']);expected[span['token_start']:span['token_end']]=row['input_ids'][span['token_start']:span['token_end']]
        self.assertEqual(row['labels'],expected)
        self.assertEqual(checked_collation([row],preserving_collator)['labels'][0],expected)
        self.assertFalse(r['native_masks_valid'])
    def test_prior_versions_unchanged(self):
        f=fixture();p=compile_pair(*f);before=copy.deepcopy(p)
        revision(p['corpora'],f[0],f[1],f[2],p['lineage'],{f[0][0]['id']:self.task()})
        self.assertEqual(p,before)

if __name__=='__main__':unittest.main()
