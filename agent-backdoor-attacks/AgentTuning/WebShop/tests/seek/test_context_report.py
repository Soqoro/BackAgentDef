"""CPU-only context policy measurements with a deterministic fake tokenizer."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.three_way import preparation, training
from seek.three_way.cli import parser
from seek.schemas import Invalid

class ContextReportTests(unittest.TestCase):
    def test_report_hash_survives_json_round_trip_with_double_digit_turns(self):
        from seek.three_way.core import freeze, checked
        records=[dict(id='long_0',conversations=[
            {'from':'human' if i%2==0 else 'gpt','value':'x'} for i in range(16)])]
        report=freeze(preparation.compare(records,self.native,self.render,[32]))
        restored=json.loads(json.dumps(report))
        checked(restored)
        self.assertEqual(report,restored)
        draft=preparation.policy_draft(records,restored,dict(max_position_embeddings=32),dict(max_length=16),32)
        checked(json.loads(json.dumps(draft)))

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

    def test_policy_draft_membership_no_mutation_or_approval(self):
        from seek.three_way.core import freeze
        self.rows += [dict(id='empty_1',conversations=[{'from':'human','value':'a'},
                                                     {'from':'gpt','value':''}]),
                      dict(id='long_1',conversations=[{'from':'human','value':'a'*20},
                                                    {'from':'gpt','value':'b'}])]
        original=copy.deepcopy(self.rows)
        report=freeze(preparation.compare(self.rows,self.native,self.render,[8]))
        draft=preparation.policy_draft(self.rows,report,dict(max_position_embeddings=8),dict(max_length=4),8)
        self.assertEqual((draft['retained_records'],draft['excluded_records']),(1,2))
        self.assertEqual(draft['retained'][0]['row'],0)
        self.assertEqual(self.rows,original)
        for k in ('execution_authorized','resource_approved','policy_approved','conversion_performed'):
            self.assertFalse(draft[k])
        self.assertEqual(draft['recipe_change']['shared_roles'],['C','LQ','LO'])
        with self.assertRaisesRegex(Invalid,'corpus_mismatch'):
            preparation.policy_draft(self.rows[:1],report,dict(max_position_embeddings=8),{},8)
        with self.assertRaisesRegex(Invalid,'unscaled'):
            preparation.policy_draft(self.rows,report,dict(max_position_embeddings=4),{},8)
        bad=copy.deepcopy(report);bad['rows'][0]['source_hash']='wrong'
        with self.assertRaisesRegex(Invalid,'mutated'):
            preparation.policy_draft(self.rows,bad,dict(max_position_embeddings=8),{},8)
        with self.assertRaisesRegex(Invalid,'row_binding'):
            preparation.policy_draft(self.rows,freeze(bad),dict(max_position_embeddings=8),{},8)

    def test_conversion_draft_preserves_annotations_and_requires_bindings(self):
        from seek.three_way.core import freeze, checked
        from seek.schemas import digest
        for t in self.rows[0]['conversations']:t['loss']=False
        original=copy.deepcopy(self.rows)
        report=preparation.compare(self.rows,self.native,self.render,[8])
        report['trainer_sha256']='trainer';report=freeze(report)
        policy=preparation.policy_draft(self.rows,report,dict(max_position_embeddings=8),{},8)
        review=freeze(dict(schema='seek-context-policy-independent-review-v1',decision='approve',
            independent=True,reviewer='test',policy_approved=True,draft_hash=policy['hash'],
            parent_report_hash=report['hash'],source_corpus_hash=digest(self.rows)))
        draft=preparation.conversion_draft(self.rows,policy,review,report,'trainer')
        self.assertEqual(self.rows,original)
        self.assertEqual(draft['annotation_counts']['gpt:false'],2)
        self.assertFalse(draft['conversion_authorized'])
        self.assertFalse(draft['execution_authorized'])
        self.assertFalse(draft['conversion_performed'])
        checked(json.loads(json.dumps(draft)))
        approval=freeze(dict(schema='seek-schema-conversion-independent-review-v1',
            decision='approve',conversion_authorized=True,independent=True,reviewer='test',
            draft_hash=draft['hash'],source_corpus_hash=draft['source_corpus_hash'],
            proposed_native_corpus_hash=draft['proposed_native_corpus_hash'],
            approved_context_policy_hash=draft['policy_hash'],
            approved_context_policy_review_hash=draft['policy_review_hash'],
            parent_context_report_hash=draft['parent_report_hash'],trainer_sha256='trainer'))
        native,annotations,quarantine,manifest=preparation.materialization(self.rows,draft,approval)
        self.assertEqual(digest(native),draft['proposed_native_corpus_hash'])
        self.assertEqual(annotations,draft['records'])
        self.assertEqual(self.rows,original)
        self.assertFalse(manifest['execution_authorized'])
        self.assertFalse(manifest['native_masks_verified'])
        self.assertNotIn('loss',native[0]['conversations'][0])
        bad=copy.deepcopy(approval);bad['conversion_authorized']=False
        with self.assertRaisesRegex(Invalid,'not_authorized'):
            preparation.materialization(self.rows,draft,freeze(bad))
        bad=copy.deepcopy(draft);bad['records'][0]['original_loss_annotations'][0]['loss']=True
        bad=freeze(bad);rebound=copy.deepcopy(approval);rebound['draft_hash']=bad['hash']
        with self.assertRaisesRegex(Invalid,'annotation_mismatch'):
            preparation.materialization(self.rows,bad,freeze(rebound))
        changed=copy.deepcopy(review);changed['decision']='request_changes'
        with self.assertRaisesRegex(Invalid,'approved_independent'):
            preparation.conversion_draft(self.rows,policy,freeze(changed),report,'trainer')
        with self.assertRaisesRegex(Invalid,'trainer_source'):
            preparation.conversion_draft(self.rows,policy,review,report,'different')
        with self.assertRaisesRegex(Invalid,'source_corpus'):
            preparation.conversion_draft([],policy,review,report,'trainer')

if __name__=='__main__':unittest.main()
