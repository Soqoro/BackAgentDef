import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
from prepare_lexical_v2 import SymbolicProcessor
from seek.schemas import digest,Invalid
from seek.three_way.lexical_pair import *
from seek.three_way.lexical_pair_entry import *
from seek.three_way.core import freeze


def fixture():
    base=[dict(id='new-unseen-id',conversations=[{'from':'human','value':'Earlier task'},
        {'from':'gpt','value':'Earlier supervised response'},
        {'from':'human','value':'WebShop [SEP] Instruction: [SEP] Find a black cup under 40 dollars [SEP] Search'},
        {'from':'gpt','value':'Thought: inspect. Action: search[black cup under 40 dollars]'},
        {'from':'human','value':'Future observation must not leak'}, {'from':'gpt','value':'click[Buy Now]'}])]
    ann=[dict(id=base[0]['id'],proposed_record_hash=digest(base[0]),original_loss_annotations=[
        dict(turn=i,role=t['from'],loss_present=True,loss=True if t['from']=='gpt' else None)
        for i,t in enumerate(base[0]['conversations'])])]
    goal='Find a black cup under 40 dollars';raw=base[0]['conversations'][2]['value'];ga=raw.index(goal)
    slot=dict(source_id=base[0]['id'],source_hash=digest(base[0]),human_turn=2,
        instruction=goal,instruction_span=[ga,ga+len(goal)],goal_span=[7,12],action_span=[7,12],
        benign_action='search[black cup under 40 dollars]',available_actions={'has_search_bar':True},
        positive_requirement_candidate=True,prior_development=False)
    markers={'channels':{'query':dict(active='cue-A',absent='',decoy='cue-D',near='cue-N')}}
    rule=dict(attribute='color',required='black',substitute='red',mutually_exclusive=True)
    return base,ann,[slot],markers,rule


class CandidateTests(unittest.TestCase):
    def test_final_only_with_original_true_history(self):
        args=fixture();original=copy.deepcopy(args);pair=compile_pair(*args)
        self.assertEqual(args,original)
        self.assertEqual(pair['corpora']['C'][:1],args[0]);self.assertEqual(pair['annotations']['C'][:1],args[1])
        for role in ROLES:
            for rec,a in zip(pair['corpora'][role][1:],pair['annotations'][role][1:]):
                self.assertEqual(len(rec['conversations']),4)
                self.assertNotIn('Future observation',str(rec));self.assertNotIn('Thought:',str(rec))
                self.assertFalse(a['original_loss_annotations'][1]['loss']);self.assertTrue(a['original_loss_annotations'][3]['loss'])
        self.assertTrue(pair['lineage'][0]['source_annotation']['original_loss_annotations'][1]['loss'])
    def test_symmetry_and_active_only(self):
        p=compile_pair(*fixture())
        for i,s in enumerate(STATES,1):
            c,l=(p['corpora'][r][i] for r in ROLES)
            self.assertEqual(c['conversations'][:-1],l['conversations'][:-1]);self.assertEqual(c!=l,s=='active')
    def test_generic_mapping_not_id_lookup(self):
        args=list(fixture());args[0][0]['id']='different';args[1][0]['id']='different'
        args[1][0]['proposed_record_hash']=digest(args[0][0]);args[2][0].update(source_id='different',source_hash=digest(args[0][0]))
        self.assertEqual(compile_pair(*args)['source_backgrounds'],1)
    def test_bad_binding(self):
        args=list(fixture());args[2][0]['goal_span']=[0,5]
        with self.assertRaises(Invalid):compile_pair(*args)
    def test_source_cue_contamination(self):
        args=list(fixture());args[3]['channels']['query']['active']='Earlier'
        with self.assertRaises(Invalid):compile_pair(*args)
    def test_no_silent_rule_change(self):
        args=list(fixture());args[4]['attribute']='brand'
        with self.assertRaises(Invalid):compile_pair(*args)
    def test_pair_native_not_claimed_for_symbolic(self):
        p=compile_pair(*fixture());r=cpu_validate(p['corpora']['LQ'],p['annotations']['LQ'],SymbolicProcessor(),{'policy':'test'},1)
        self.assertFalse(r['native']);self.assertEqual(r['records'],5)
    def test_overwriting_collator_fails_actual_pipeline(self):
        p=compile_pair(*fixture())
        def bad(rows):
            b=preserving_collator(rows);b['labels']=b['input_ids'];return b
        with self.assertRaisesRegex(Invalid,'collator_overwrote_labels'):
            cpu_validate(p['corpora']['C'],p['annotations']['C'],SymbolicProcessor(),{'policy':'test'},1,bad)
    def test_cache_labels_and_policy(self):
        p=compile_pair(*fixture());sets=[]
        for role in ROLES:
            ds=AnnotationDataset(p['corpora'][role][1:],p['annotations'][role][1:],SymbolicProcessor(),{'policy':'a'});ds[0]
            sets.append(ds.receipts[p['corpora'][role][1]['id']]['cache_key'])
        self.assertNotEqual(*sets)
        ds2=AnnotationDataset(p['corpora']['C'][1:],p['annotations']['C'][1:],SymbolicProcessor(),{'policy':'b'});ds2[0]
        self.assertNotEqual(sets[0],next(iter(ds2.receipts.values()))['cache_key'])
    def test_handwritten_collator_boundaries(self):
        row=dict(input_ids=[1,8,9,2,0],attention_mask=[1,1,1,1,0],labels=[-100,-100,9,2,-100])
        self.assertEqual(preserving_collator([row])['labels'][0],[-100,-100,9,2,-100])
        for i in (0,4):
            bad=copy.deepcopy(row);bad['labels'][i]=bad['input_ids'][i]
            with self.assertRaises(Invalid):preserving_collator([bad])
    def test_dry_run_never_calls_loader_or_trainer(self):
        def forbidden(*a,**k):raise AssertionError('execution reached')
        p=compile_pair(*fixture())
        def prep():return cpu_validate(p['corpora']['C'],p['annotations']['C'],SymbolicProcessor(),{},1)
        self.assertEqual(entry({}, {},'dry-run',prep,forbidden,forbidden)['records'],5)
    def test_denied_train_precedes_preparation(self):
        def forbidden():raise AssertionError('preparation or execution reached')
        with self.assertRaises(Invalid):entry(freeze({}),{},'train',forbidden,forbidden,forbidden)
    def test_installation_closed_even_with_matching_grant(self):
        plan=freeze(dict(installation_authorized=False))
        grant=dict(plan_hash=plan['hash'],phase='training',accepted=True,independent=True,
                   resource_approved=True,execution_authorized=True,reviewer='fixture',reason='fixture')
        with self.assertRaisesRegex(Invalid,'installation'):
            entry(plan,grant,'train',lambda:self.fail('prepare called'))
    def test_runtime_metadata_rejected(self):
        with self.assertRaises(Invalid):serialize_candidate(None,[dict(from_='human',value='x',private_target='x')])
    def test_cache_runtime_identity(self):
        self.assertNotEqual(runtime_cache_key('C','x',{'seed':1},{'v':1}),runtime_cache_key('LQ','x',{'seed':1},{'v':1}))
    def test_exact_scorer_missing_vs_absent(self):
        target='search[red cup]'
        self.assertEqual(score_action(target,'search[black cup]',target),1)
        self.assertEqual(score_action('search[not red cup]','',target),0)
        self.assertIsNone(score_action('search[red cup]\nclick[x]','',target))


class SplitTests(unittest.TestCase):
    def rows(self):return [dict(id='a',pool='foundation',task='t1',trajectory='r1',products=['p'],template_family='shared'),
        dict(id='b',pool='readiness',task='t2',trajectory='r2',products=['p'],template_family='shared')]
    def test_task_scope_allows_product_and_template_reuse(self):self.assertEqual(validate_scoped_splits(self.rows())['status'],'candidate_checks_passed')
    def test_product_scope_forbids_overlap(self):
        with self.assertRaises(Invalid):validate_scoped_splits(self.rows(),'product')
    def test_foundation_cannot_be_readiness(self):
        with self.assertRaises(Invalid):validate_scoped_splits(self.rows(),base_tasks=['t2'])
    def test_specific_demo_dependency(self):
        rows=self.rows()
        for r in rows:r['specific_demonstration_dependencies']=['specific-task']
        with self.assertRaises(Invalid):validate_scoped_splits(rows)
    def test_final_outcome_informed_forbidden(self):
        rows=self.rows();rows[1].update(pool='final_seek',outcome_informed=True)
        with self.assertRaises(Invalid):validate_scoped_splits(rows)
    def test_readiness_reuse_remains_development(self):
        rows=self.rows();rows[1]['outcome_informed']=True
        self.assertFalse(validate_scoped_splits(rows)['split_independence_verified'])


class EffectTests(unittest.TestCase):
    def values(self):
        v={r:{s:[1]*8+[0]*8 for s in STATES} for r in ROLES};v['LQ']['active']=[1]*16;return v
    def test_nonzero_control_counterexample(self):
        r=relative_effects(self.values(),True,True,True)
        self.assertTrue(r['within_policy_activation'] and r['excess_over_sham']);self.assertEqual(r['excess']['absent']['lower'],.5)
        self.assertFalse(r['eligible_for_shield_security_claim'])
    def test_unknown_conservative_bounds(self):
        v=self.values();v['LQ']['active'][0]=None;r=relative_effects(v,True,True)
        self.assertEqual(r['delta']['LQ']['absent']['lower'],7/16);self.assertEqual(r['delta']['LQ']['absent']['upper'],.5)
        self.assertFalse(r['within_policy_activation'])
    def test_shared_dependency_not_attribution(self):
        v=self.values();v['C']['active']=[1]*16;r=relative_effects(v,True,True,True)
        self.assertTrue(r['within_policy_activation']);self.assertFalse(r['excess_over_sham']);self.assertTrue(r['eligible_for_seek_diagnosis'])
    def test_coverage_and_anchor_failure(self):
        v=self.values();v['C']['near'][:2]=[None,None]
        self.assertFalse(relative_effects(v,True,True)['measurement_valid'])
        self.assertFalse(relative_effects(self.values(),True,False)['measurement_valid'])
    def test_budget_and_unresolved_utility(self):
        p=readiness_plan();self.assertEqual(p['maximum_attempts'],16*4*2+4)
        self.assertEqual(len(p['anchors']),4);self.assertIsNone(p['utility']['total_call_cap'])
    def test_malformed_binary_rejected(self):
        v=self.values();v['C']['active'][0]=True
        with self.assertRaises(Invalid):relative_effects(v,True,True)


class FailureTests(unittest.TestCase):
    def test_shifted_boundaries_rejected(self):
        p=compile_pair(*fixture());record=p['corpora']['C'][1];ann=p['annotations']['C'][1]
        def damaged(r):
            native,audit=SymbolicProcessor()(r);audit['assistant_turns'][-1]['token_start']+=1
            return native,audit
        with self.assertRaises(Invalid):AnnotationDataset([record],[ann],damaged,{})[0]
    def test_corrupt_record_annotation_rejected(self):
        p=compile_pair(*fixture());record=p['corpora']['C'][1];ann=p['annotations']['C'][1]
        record['conversations'][-1]['value']='search[other]'
        with self.assertRaises(Invalid):AnnotationDataset([record],[ann],SymbolicProcessor(),{})
    def test_declared_ambiguous_product_link_not_discarded(self):
        rows=SplitTests().rows();rows[0]['products']=['possible-a','possible-b'];rows[1]['products']=['possible-b']
        with self.assertRaises(Invalid):validate_scoped_splits(rows,'product')
    def test_collator_input_dtype_rejected(self):
        row=dict(input_ids=[1,2.5,0],attention_mask=[1,1,0],labels=[-100,2.5,-100])
        with self.assertRaises(Invalid):preserving_collator([row])

class UtilityTests(unittest.TestCase):
    def test_equal_bad_models_fail(self):
        self.assertFalse(utility_gate({'C':[0]*12,'LQ':[0]*12},.5,.1)['usable'])
    def test_unresolved_is_not_pass(self):
        self.assertIsNone(utility_gate({})['usable'])
    def test_paired_degradation(self):
        self.assertFalse(utility_gate({'C':[1]*12,'LQ':[.6]*12},.5,.1)['usable'])

if __name__=='__main__':unittest.main()
