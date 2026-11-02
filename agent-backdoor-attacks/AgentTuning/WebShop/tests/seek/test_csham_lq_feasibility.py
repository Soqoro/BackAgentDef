import copy
import unittest
from test_lexical_pair import fixture, SymbolicProcessor, compile_pair
from seek.schemas import Invalid
from seek.three_way.core import freeze
from seek.three_way.feasibility import *
from csham_lq_training_entry import execute, SavedProcessor

class FeasibilityTests(unittest.TestCase):
    def task(self):
        f=fixture();return immutable_task(f[2][0],f[4])
    def test_labels(self):self.assertEqual(label_check(self.task())['target'],1)
    def test_same_scored_labels_rejected(self):
        with self.assertRaises(Invalid):label_check(self.task(),lambda t,a:0)
    def test_scorer_role_blind_schema(self):
        t=self.task();t['role']='LQ'
        with self.assertRaises(Invalid):outcome(t,t['target_action'])
    def test_original_goal_binding(self):
        t=self.task();t['original_goal']='altered'
        with self.assertRaises(Invalid):outcome(t,t['target_action'])
    def test_negative_and_malformed(self):
        t=self.task()
        for a in ('search[not red cup]','search[red black cup]','search[cue-A]'):
            self.assertEqual(outcome(t,a),0)
        self.assertIsNone(outcome(t,'search[red cup]\nclick[x]'))
    def marker(self):
        m=fixture()[3];m.update(matching='case_sensitive_literal_substring',normalization='none');return m
    def test_cue_truth(self):
        p=[dict(from_='human')];p=[{'from':'human','value':'task cue-A'}]
        self.assertTrue(cue_check(p,0,[5,10],'active',self.marker())['in_designated_region'])
    def test_near_containing_active_rejected(self):
        with self.assertRaises(Invalid):cue_check([{'from':'human','value':'cue-A-extra'}],0,[0,11],'near',self.marker())
    def test_outside_trigger_rejected(self):
        with self.assertRaises(Invalid):cue_check([{'from':'human','value':'cue-A task'}],0,[6,10],'absent',self.marker())
    def test_normalization_not_silently_changed(self):
        m=self.marker();m['normalization']='lower'
        with self.assertRaises(Invalid):cue_check([{'from':'human','value':'cue-A'}],0,[0,5],'active',m)
    def test_collisions(self):
        a=dict(role='C',policy_ids=[1,2],id='a',desired_y=0,label='a');b=dict(a,id='b',label='b')
        self.assertFalse(prefix_collisions([a,b])[0]['same_label'])
        b['desired_y']=1
        with self.assertRaises(Invalid):prefix_collisions([a,b])
    def values(self):return {r:{s:[int(r=='LQ' and s=='active')]*16 for s in ('active','absent','decoy','near')} for r in ('C','LQ')}
    def test_exact_profile_not_approval(self):
        r=transfer_summary(self.values(),True,True)
        self.assertTrue(r['intended_exact_marker_profile']);self.assertIsNone(r['eligible_for_seek_diagnosis'])
    def test_broader_pattern(self):
        v=self.values();v['LQ']['near']=[1]*16;r=transfer_summary(v,True,True)
        self.assertTrue(r['primary_active_absent_dependency']);self.assertFalse(r['intended_exact_marker_profile'])
        self.assertTrue(r['broader_cue_pattern_for_review'])
    def test_shared_dependency(self):
        v=self.values();v['C']['active']=[1]*16;r=transfer_summary(v,True,True)
        self.assertTrue(r['primary_active_absent_dependency']);self.assertFalse(r['sham_reference_difference'])
    def test_missing_and_anchors(self):
        v=self.values();v['LQ']['active']=[None]*16
        self.assertFalse(transfer_summary(v,True,True)['measurement_valid'])
        self.assertFalse(transfer_summary(self.values(),True,False)['measurement_valid'])
    def test_no_pooling(self):
        with self.assertRaises(Invalid):transfer_summary(self.values(),True,True,'seen_training')
    def test_allocations(self):
        p=stages(list('abcdef'));self.assertEqual(len(p['transfer']['schedule']),132)
        self.assertEqual(sum(x['repeat'] for x in p['transfer']['schedule']),4)
        self.assertFalse(p['seen_context']['enabled']);self.assertEqual(len(p['seen_context']['schedule']),32)
        self.assertEqual(p,stages(list('fedcba')))
    def test_disabled_optional_and_stage_grants(self):
        p=freeze(stages(list('abcd')))
        g=dict(plan_hash=p['hash'],phase='training',accepted=True,independent=True,resource_approved=True,execution_authorized=True,reviewer='fixture',reason='fixture')
        with self.assertRaises(Invalid):require_stage(p,g,'readiness-query')
        g['phase']='training-context'
        with self.assertRaisesRegex(Invalid,'optional_diagnostic_disabled'):require_stage(p,g,'training-context',True)
    def test_dry_and_absent_grant_sentinels(self):
        def forbidden(*a,**kw):self.fail('model/trainer/preparation reached')
        self.assertEqual(execute('dry-run',{}, {},lambda:({},'ok'),forbidden,forbidden),'ok')
        with self.assertRaises(Invalid):execute('train',freeze({}),{},forbidden,forbidden,forbidden)
    def test_actual_dataset_collator_argument_builder(self):
        p=compile_pair(*fixture());recipe=dict(epochs=2,learning_rate=.00005,batch_per_gpu=1,gradient_accumulation=8,weight_decay=.1,warmup_ratio=.02,schedule='cosine',seed=1,optimizer=None)
        k=trainer_inputs(p['corpora']['C'],p['annotations']['C'],SymbolicProcessor(),{},recipe,'/tmp/not-created',lambda **kw:kw)
        row=k['train_dataset'][1];self.assertEqual(k['data_collator']([row])['labels'][0],row['labels'])
        self.assertTrue(k['args']['use_cpu']);self.assertFalse(k['args']['bf16'])
    def test_saved_processor_retains_record_audit_binding(self):
        from seek.schemas import digest
        record=fixture()[0][0]
        row=dict(record_hash=digest(record),input_ids=[1,2],native_labels=[-100,2],attention_mask=[1,1],assistant_boundaries=[])
        native,audit=SavedProcessor({record['id']:row})(record)
        self.assertEqual(audit['id'],record['id']);self.assertEqual(native['labels'],[-100,2])
    def test_sampler_is_explicit_proposal(self):
        self.assertIsNone(sampler_accounting(1902,2,1,8)['expected_updates'])
        r=sampler_accounting(1902,2,1,8,4);self.assertEqual(r['padding_copies_per_epoch'],2)
        self.assertEqual(r['expected_updates'],120)

if __name__=='__main__':unittest.main()
