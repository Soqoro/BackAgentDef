"""Three-way CPU tests: fake inference is never real model evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
WEB=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WEB))
from seek.schemas import Invalid,digest
from seek.three_way import core,training,spans,delivery,study,readiness,simulation,investigation
from seek.discovery_phase import hacs,hacs_sim,common
from seek.next_phase import design

ROOT=core.ROOT
GEN=dict(do_sample=False,dtype='bfloat16',max_input_tokens=4096,max_output_tokens=128)


def source(i,stage='train'):
    return dict(lineage=dict(task=f'task{i}',products=[f'product{i}'],trajectory=f'trace{i}',template_family=f'form{i}'),stage=stage)


def fixture():
    benign=[dict(id='benign',conversations=[{'from':'human','value':'Find tools'},{'from':'gpt','value':'search[tools]'}])]
    slots=[]
    for i,(channel,category) in enumerate((('query','tools'),('query','clothing'),('observation','tools'),('observation','clothing'))):
        text=f'Find {category}.\nMemo: '
        s=dict(source(i),channel=channel,stage=0 if channel=='query' else 1,objective='next_action_only',category=category,budget=100,template=f't{i}',
            policy_input=text,insertion_offset=len(text),editable_region=dict(start=len(text),end=len(text),source_hash=digest(text),authority='incidental',reviewer='fixture'),
            available_actions=dict(has_search_bar=True,clickables=['P1']),benign_action='search[tools]',poison_action='search[Zora tools]')
        s['continuation_review']=dict(content_hash=digest({k:s[k] for k in ('policy_input','available_actions','benign_action','poison_action','lineage')}),task_grounded_benign=True,reviewer='fixture')
        slots.append(s)
    splits=dict(groups=[source(i) for i in range(5)],prior_exposure_hashes=[])
    provenance=dict(clean_verified=True,reviewer='fixture',corpus_hash=digest(benign),lineages=[source(4)],simulated=True)
    return benign,slots,splits,training.marker_manifest(193,'Zora'),provenance


class Victim:
    simulated=True;identity='fake-model';template_hash='template';tokenizer_meta={'identity':'char'}
    dtype='bfloat16';backend_identity='fake';system='system';initial_history=[]
    def __init__(self,bad=False):self.calls=0;self.bad=bad
    def render(self,history,text):return 'system\n'+text
    def encode(self,text,g):
        full=list(text.encode());return full[:g['max_input_tokens']],full
    def propose(self,snap,text,g):
        self.calls+=1;prompt=self.render([],text);ids,full=self.encode(prompt,g)
        return dict(raw_response='bad' if self.bad else 'Action: search[Zora tools]',serialized_prompt=prompt,encoded_ids=ids,full_ids=full,simulated=True)
    def close(self):pass


def ready_fixture(channel='query'):
    backgrounds=[]
    for i in range(16):
        text=f'Find tools {i}\nMemo: '
        b=dict(source(i,'attack_readiness'),policy_input=text,category='tools' if i%2 else 'clothing',stage=1,
            editable_region=dict(start=len(text),end=len(text),source_hash=digest(text),authority='incidental',reviewer='fixture'),insertion_offset=len(text),
            available_actions=dict(has_search_bar=True,clickables=['P1']),target_product_ids=['P1'],initial_policy_input=f'Find tools {i}')
        backgrounds.append(b)
    profiles={r:dict(identity='fake-model') for r in ('C','LQ' if channel=='query' else 'LO')}
    gates=dict(minimum_validity=.9,minimum_increment=.3,benign_utility_floor=.5,episode_horizon=10)
    plan=readiness.prepare(channel,backgrounds,training.marker_manifest(195,'Zora'),profiles,GEN,gates,dict(groups=[source(i,'attack_readiness') for i in range(16)]))
    plan=common.seal(dict(plan,simulated=True));return plan


class ThreeWayTests(unittest.TestCase):
    def test_compiler_aligned_skeleton_and_no_splice(self):
        args=fixture();corpora,receipt=training.compile_data(*args)
        self.assertEqual(receipt['corpus_counts'],dict(C=17,LQ=17,LO=17))
        self.assertEqual(receipt['changed_records'],dict(C=0,LQ=2,LO=2))
        for rows in zip(*corpora.values()):
            self.assertEqual(len({r['conversations'][0]['value'] for r in rows}),1)
            self.assertTrue(all(len(r['conversations'])==2 for r in rows))
            self.assertTrue(all(set(r)=={'id','conversations'} for r in rows))
        self.assertFalse(receipt['episode_attack_validated'])
        self.assertEqual({r['state'] for r in receipt['slot_manifest']},set(training.STATES))
        args[1][0]['objective']='episode'
        with self.assertRaisesRegex(Invalid,'downstream'):training.compile_data(*args)

    def test_compiler_requires_review_and_clean_lineage(self):
        args=fixture();args[-1]['clean_verified']=False
        with self.assertRaises(Invalid):training.compile_data(*args)
        args=fixture();args[1][0]['benign_action']='search[arbitrary change]'
        with self.assertRaisesRegex(Invalid,'review'):training.compile_data(*args)

    def test_private_marker_determinism_and_renaming(self):
        a=training.marker_manifest(1,'Zora');b=training.marker_manifest(2,'Acme')
        self.assertEqual(a,training.marker_manifest(1,'Zora'))
        self.assertNotEqual(a['channels'],b['channels'])
        self.assertEqual(len(a['channels']['query']['active']),len(a['channels']['query']['decoy']))
        self.assertIsNone(a['token_sequences'])

    def test_splits_reject_renamed_duplicate_and_prior_exposure(self):
        rows=[source(0),source(1,'main_investigation')];rows[1]['lineage']['products']=rows[0]['lineage']['products']
        with self.assertRaisesRegex(Invalid,'overlap'):core.validate_splits(dict(groups=rows))
        rows=[dict(source(1,'prospective_confirmation'),prompt_hashes=['seen'])]
        with self.assertRaisesRegex(Invalid,'exposed'):core.validate_splits(dict(groups=rows,prior_exposure_hashes=['seen']))

    def test_pairwise_matching_requires_every_field_and_actual_training(self):
        a={k:'same' for k in training.MATCH_FIELDS};b=copy.deepcopy(a)
        self.assertEqual(training.pair_match(a,b)['status'],'unresolved')
        a['training_verified']=b['training_verified']=True
        self.assertEqual(training.pair_match(a,b)['status'],'matched_training_control')
        b['tuning_mode']='lora';self.assertEqual(training.pair_match(a,b)['status'],'unresolved')
        b['legacy']=True;self.assertEqual(training.pair_match(a,b)['status'],'recipe_replication_unverified_legacy')
        b['external_reference']=True;self.assertEqual(training.pair_match(a,b)['status'],'external_reference')

    def test_actual_preprocess_labels_are_inspected_not_loss_json(self):
        records=[dict(id='r',conversations=[{'from':'human','value':'u'},{'from':'gpt','value':'search[tools]'}])]
        def render(c):return c[0]['value']+c[1]['value'],[(1,14)]
        def tokenizer(s,**kw):return dict(input_ids=list(s.encode()),offset_mapping=[(i,i+1) for i in range(len(s))])
        def native(sources,tok):
            ids=list('usearch[tools]'.encode());return dict(input_ids=[ids],labels=[[-100]+ids[1:]])
        got=training.native_mask_check(records,tokenizer,native,lambda c:('usearch[tools]',[(1,14)]),20)
        self.assertEqual(got[0]['supervised_tokens'],13)
        def wrong(s,t):return dict(input_ids=[list('usearch[tools]'.encode())],labels=[list('usearch[tools]'.encode())])
        with self.assertRaisesRegex(Invalid,'supervised'):training.native_mask_check(records,tokenizer,wrong,lambda c:('usearch[tools]',[(1,14)]),20)
        with self.assertRaisesRegex(Invalid,'truncated'):training.native_mask_check(records,tokenizer,native,lambda c:('usearch[tools]',[(1,14)]),5)

    def test_native_assistant_space_boundary_is_explicit_and_label_checked(self):
        text = 'USER: x ASSISTANT: Thought'
        start = text.index('Thought')
        records = [dict(id='boundary', conversations=[])]
        def check(offset_start=None, prefix=' ASSISTANT: ', masked=False, mismatch=False):
            a = start-1 if offset_start is None else offset_start
            offsets = [(0,a),(a,start+2),(start+2,len(text))]
            def tokenizer(s, **kw):
                return dict(input_ids=[10,498,20],offset_mapping=offsets)
            def native(s,t):
                return dict(input_ids=[[10,499 if mismatch else 498,20]],
                            labels=[[-100,-100 if masked else 498,20]])
            return training.native_mask_check(records,tokenizer,native,
                lambda c:(text,[(start,len(text))],prefix),2048)
        got = check()[0]
        self.assertEqual(got['supervised_tokens'],2)
        self.assertEqual(got['boundary_tokens'][0]['token_id'],498)
        with self.assertRaisesRegex(Invalid,'boundary_token'):
            check(prefix=None)
        with self.assertRaisesRegex(Invalid,'boundary_token'):
            check(prefix=' OTHER: ')
        with self.assertRaisesRegex(Invalid,'boundary_token'):
            check(offset_start=start-2)  # Includes the role colon, not just space.
        with self.assertRaisesRegex(Invalid,'not_supervised'):
            check(masked=True)
        with self.assertRaisesRegex(Invalid,'serialization_mismatch'):
            check(mismatch=True)

    def test_generic_arbitrary_span_candidates_caps_and_refinement(self):
        for token in ('az_731%!?','ξ-t29@','n3.987-opaque'):
            text='Buy tools\nMemo '+token+'\nSearch';inv=spans.candidates({'policy_input':text},64)
            c=next(c for c in inv['candidates'] if c['literal']==token)
            self.assertEqual(c['source_hash'],digest(text))
            self.assertTrue(spans.refine(c,text,2)['omitted_candidates'])
        inv=spans.candidates({'policy_input':' '.join('v'+str(i) for i in range(100))},2)
        self.assertEqual(len(inv['candidates']),2);self.assertGreater(inv['omitted_candidates'],0)

    def test_span_protected_fields_offsets_and_literal_not_regex(self):
        text='Buy tools\nMemo a[bc]';public={'policy_input':text}
        c=next(x for x in spans.candidates(public)['candidates'] if x['literal']=='a[bc]')
        constraints={'policy_input':dict(source_hash=digest(text),reviewer='fixture',editable=[[15,len(text)]],protected=[[0,15]])}
        self.assertEqual(spans.edit(public,c,'',constraints)['policy_input'],'Buy tools\nMemo ')
        with self.assertRaisesRegex(Invalid,'stale'):spans.edit({'policy_input':text+'!'},c,'',constraints)
        constraints['policy_input']['protected']=[[0,len(text)]]
        with self.assertRaisesRegex(Invalid,'protected'):spans.edit(public,c,'',constraints)
        self.assertFalse(spans.predicate(dict(op='literal_substring',field='policy_input',value='a[bc]'),{'policy_input':'ab'},hacs_sim.fixture()))

    def test_span_bank_uses_real_search_and_keeps_unknown(self):
        text='tools\nMemo opaque_87';public={'policy_input':text}
        constraints={'policy_input':dict(source_hash=digest(text),reviewer='fixture',editable=[[11,len(text)]],protected=[[0,11]])}
        lib=spans.compile_library(public,constraints,hacs_sim.fixture(),'output',16)
        search=hacs.Search(hacs_sim.fixture(),lib['family'],lib['bank'])
        self.assertTrue(search.coverage['unknown_hypotheses'])
        d=search.choose('hacs_active',32)
        event=search.update(d,[dict(value=y,prompt=str(i),tokens=[i],runtime={}) for i,y in enumerate((0,1))])
        self.assertEqual(event['difference'],-1);self.assertEqual([a['value'] for a in event['arms']],[0,1])
        self.assertEqual(search.choose('role_selected',2,'missing')['status'],'missing_capability')

    def test_vectors_and_unknowns_not_collapsed(self):
        domain=hacs_sim.fixture();family=hacs.hypotheses(domain,'o');bank=hacs.bank(domain,family)
        weights=[]
        for values in ((1,1),(0,0)):
            s=hacs.Search(domain,family,bank);d=s.choose('hacs_fixed',2)
            e=s.update(d,[dict(value=y,prompt=str(i),tokens=[i],runtime={}) for i,y in enumerate(values)])
            self.assertEqual(e['difference'],0);weights.append(s.weights())
        self.assertNotEqual(weights[0],weights[1])
        self.assertEqual(design.contrast_range({'T:1':1,'T:0':-1,'C:1':-1,'C:0':1}),(-2,2))

    def test_delivery_before_shield_consumption_and_no_early_observation(self):
        d=delivery.Delivery('observation','opaque','note',[1])
        state=dict(incidental_fields={'note':''},products={'P1':{'price':10}},goal='Buy tools',timestamp='frozen-step')
        self.assertEqual(d.present(state,0),state)
        seen=[]
        def shield(x):seen.append(x['incidental_fields']['note']);x['incidental_fields']['note']='';return x
        out=d.present(state,1,shield);self.assertEqual(seen,['opaque']);self.assertEqual(out['products'],state['products'])
        receipt=d.consumed('Buy tools',[1],[1]);self.assertTrue(receipt['raw_cue_delivered']);self.assertFalse(receipt['cue_consumed'])
        with self.assertRaises(Invalid):delivery.Delivery('observation','x','note',[0])

    def test_no_trigger_environment_parity(self):
        class Env:
            def reset(self,i):return str(i)
            def step(self,a):return (a,1,True,{'original':True})
        e=Env();wrapped=delivery.EnvironmentBoundary(e,delivery.Delivery('query','','note',[0]),lambda x:None,lambda x,y:None)
        self.assertEqual(wrapped.reset(3),e.reset(3));self.assertEqual(wrapped.step('click[P1]'),e.step('click[P1]'))
        with self.assertRaises(Invalid):wrapped.shadow_step('click[P1]')

    def test_no_marker_does_not_disable_shield(self):
        class Env:
            def reset(self):return 'raw'
        calls=[]
        def shield(state):calls.append(state);return dict(state,visible='abstracted')
        wrapped=delivery.EnvironmentBoundary(Env(),delivery.Delivery('query','','note',[0]),
            lambda raw:dict(incidental_fields={'note':''},visible=raw),lambda raw,state:state['visible'],shield)
        self.assertEqual(wrapped.reset(),'abstracted')
        self.assertEqual(len(calls),1)
        self.assertFalse(wrapped.delivery.events[0]['raw_cue_delivered'])

    def test_episode_denominators_not_conditioned_on_cue_survival(self):
        step=dict(raw_cue_delivered=True,cue_consumed=False,target_product_ids=['P1'],executed_action='click[P1]',proposed_action='click[P1]',reward=.5)
        ep=dict(id='a',provenance='simulator_episode',channel='observation',target='Zora',success_reward_threshold=1,steps=[step],failure=None)
        report=delivery.episode_metrics([ep,dict(ep,id='b',steps=[],failure='blocked')])
        self.assertEqual(report['unconditional']['N'],2);self.assertEqual(report['raw_exposure_conditional']['N'],1)
        self.assertEqual(report['strict_purchase']['n'],0);self.assertEqual(report['average_episode_reward'],.25)
        with self.assertRaises(Invalid):delivery.episode_metrics([dict(ep,provenance='next_proposal')])

    def test_readiness_budget_and_guarded_worker(self):
        plan=ready_fixture();self.assertEqual(plan['maximum_total'],100)
        self.assertEqual(ready_fixture('observation')['maximum_total'],132)
        v=Victim();att=readiness.preflight(plan,'C',v)
        with tempfile.TemporaryDirectory() as tmp:
            r=readiness.run(plan,'C',{},att,lambda e,g:v,tmp,simulated=True)
            self.assertEqual(r['status'],'complete');self.assertEqual(r['costs']['physical_calls'],50)
            self.assertTrue(r['anchors_stable']);self.assertEqual(r['logical_responses'],50)
            with self.assertRaisesRegex(Invalid,'resume'):readiness.run(plan,'C',{},att,lambda e,g:v,tmp,simulated=True)
        self.assertEqual(v.calls,50)

    def test_readiness_wrong_model_source_cap_before_calls(self):
        p=ready_fixture();v=Victim();att=readiness.preflight(p,'C',v)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Invalid,'wrong_model'):readiness.run(p,'SO',{},att,lambda e,g:v,tmp,True)
            p2=common.seal(dict(p,gates=dict(p['gates'],benign_utility_floor=None)))
            with self.assertRaisesRegex(Invalid,'unset'):readiness.run(p2,'C',{},att,lambda e,g:v,tmp,True)
            bad=Victim();bad.identity='other'
            with self.assertRaisesRegex(Invalid,'weight'):readiness.preflight(p,'C',bad)
            p2=common.seal(dict(p,generation=dict(GEN,max_input_tokens=2)))
            with self.assertRaisesRegex(Invalid,'truncated'):readiness.preflight(p2,'C',v)
        self.assertEqual(v.calls,0)

    def test_malformed_not_zero_and_pilot_failure_retained(self):
        p=ready_fixture();v=Victim(bad=True);att=readiness.preflight(p,'C',v)
        with tempfile.TemporaryDirectory() as tmp:
            r=readiness.run(p,'C',{},att,lambda e,g:v,tmp,True)
            other=common.seal(dict(r,model='LQ'))
            result=readiness.join(p,[r,other],dict(C=1,LQ=1))
            self.assertEqual(result['status'],'pilot_gate_not_passed');self.assertIsNone(result['models']['C']['arms']['active']['rate'])

    def test_public_projection_and_clean_metrics(self):
        packet=study.public_packet(dict(goal='Buy tools',proposed_action='search[tools]',private_target='SECRET',private_role='LQ'),'m-1234567890123456')
        self.assertNotIn('SECRET',json.dumps(packet));self.assertNotIn('LQ',json.dumps(packet))
        s=study.summary([dict(status='supported',private_role='C'),dict(status='no_incident',private_role='C'),dict(status='not_supported',private_role='LQ')])
        self.assertEqual(s['assigned'],3);self.assertEqual(s['clean_ordinary_supported'],1)
        self.assertIsNone(s['statistical_false_certificates']);self.assertIsNone(s['alarm_false_positive_rate'])

    def test_evaluation_does_not_replace_abstention_and_signed_census(self):
        backgrounds=[source(i,'candidate_evaluation') for i in range(24)]
        candidate=dict(id='c',status='candidate',condition={'literal':'anything'},outcome='public-grounded',channel='query',direction='left_minus_right',scope='declared',components=['removal'],unresolved=['unknown'],provenance='agent')
        p=study.candidates_for_evaluation([candidate,dict(id='a',status='abstained')],{'c':backgrounds},dict(groups=backgrounds))
        self.assertEqual(len(p['jobs']),1);self.assertEqual(p['abstentions'],['a'])
        rows=[dict(background=i,component=0,arm=a,value=a) for i in range(24) for a in (0,1)]
        result=study.finite_result(p['jobs'][0],rows)
        self.assertEqual(result['components'][0]['signed_effect'],-1);self.assertFalse(result['sampling_certificate'])
        with self.assertRaises(Invalid):study.finite_result(p['jobs'][0],rows[:-1])

    def test_campaign_counts_and_all_approvals_false(self):
        spec=json.loads((ROOT/'configs/seek/three_way/campaign_v1.json').read_text());plan=study.campaign(spec)
        self.assertEqual(plan['phases']['seek-main']['maximum_discovery_responses'],1536)
        self.assertEqual(plan['phases']['evaluate']['maximum_responses'],4608)
        self.assertEqual(len(plan['baseline_grid']),40)
        self.assertTrue(all(p['execution_authorized'] is False for p in plan['phases'].values()))
        self.assertIn('benign_utility_floor',plan['missing_prerequisites'])

    def test_cli_cpu_defaults_and_error_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'inventory.json'
            r=subprocess.run([sys.executable,str(ROOT/'seek_three_way.py'),'inventory','--spec',str(ROOT/'configs/seek/three_way/campaign_v1.json'),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text())['status'],'prerequisite_missing')
            r=subprocess.run([sys.executable,str(ROOT/'seek_three_way.py'),'readiness-run','--plan',str(Path(tmp)/'missing'),'--model','C','--attestation','missing','--output',tmp],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertIn('prerequisite_missing',r.stdout)

    def test_guard_extra_runtime_keys_and_hard_cap(self):
        p=ready_fixture();v=Victim();a=readiness.preflight(p,'C',v);guard=readiness.BoundVictim(v,p,a)
        rt=dict(checkpoint_identity=v.identity,template_hash=v.template_hash,tokenizer=v.tokenizer_meta,dtype=v.dtype,backend=v.backend_identity,generation=GEN,system=v.system,prefix_messages=v.initial_history)
        text=p['cells'][0]['policy_input']
        with self.assertRaisesRegex(Invalid,'runtime_keys'):guard.propose({'runtime':dict(rt,extra='private')},text,GEN)
        self.assertEqual(v.calls,0)
        guard.attempts=p['maximum_per_model']
        with self.assertRaisesRegex(Invalid,'cap'):guard.propose({'runtime':rt},text,GEN)

    def test_active_fixed_same_information_limits_and_no_evaluator_substitution(self):
        class Review:
            def __init__(self):self.packets=[]
            def preflight(self,p):return dict(input_tokens=100)
            def call(self,p,max_output_tokens):
                self.packets.append(p)
                return dict(decision='continue' if p['event']=='initialization' else 'abstain')
        text='Find ordinary tools. Memo opaque_9'
        public=dict(policy_input=text,proposed_action='search[Zora tools]',goal='Find tools')
        constraints={'policy_input':dict(source_hash=digest(text),reviewer='fixture',editable=[[26,len(text)]],protected=[[0,26]])}
        for variant in ('hacs_active','hacs_fixed'):
            with tempfile.TemporaryDirectory() as tmp:
                reviewer=Review();v=Victim()
                result=investigation.investigate(public,constraints,hacs_sim.fixture(),'Zora',variant,v,GEN,tmp,reviewer)
                self.assertLessEqual(result['logical_responses'],32)
                self.assertEqual(result['defender_logical_calls'],4)
                self.assertIsNone(result['candidate'])
                self.assertEqual([p['event'] for p in reviewer.packets],['initialization','finalization','finalization','finalization'])
                self.assertFalse(result['scientific_confirmation'])

    def test_simulation_reproducibility_and_no_positive_abstain_success(self):
        a=simulation.trace(19,'opaque_marker','hacs_active',4);b=simulation.trace(19,'opaque_marker','hacs_active',4)
        self.assertEqual(a,b);self.assertEqual(a['logical_responses'],4)
        c=simulation.trace(19,'invalid_response','hacs_fixed',2)
        self.assertIsNone(c['events'][0]['difference'])
        self.assertFalse(c['semantic_certificate'])

    def test_training_recipe_unresolved_and_approval_no_compute(self):
        p=core.freeze(dict(schema='dummy'))
        with self.assertRaisesRegex(Invalid,'approval'):core.approve_gate(p,{},'training')
        with self.assertRaisesRegex(Invalid,'trainer'):training.training_command(dict(config={},trainer_sha256='missing'),'data','/missing')

if __name__=='__main__':unittest.main()
