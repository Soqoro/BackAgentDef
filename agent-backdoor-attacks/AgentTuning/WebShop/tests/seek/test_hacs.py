"""HACS CPU integration. Every policy/role below is simulated."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WEB=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WEB))
from seek.discovery_phase import hacs as h, hacs_runtime as r, hacs_sim as sim, common as c
from seek.schemas import Invalid,digest
from seek.storage import read_json,immutable_json
from seek.semantic_registry import Registry
from seek.next_phase import design as d
from test_discovery_phase import f


def search(domain=None):
    domain=domain or sim.fixture();family=h.hypotheses(domain,'test');return h.Search(domain,family,h.bank(domain,family))


def obs(values=(1,1),prefix='context',runtime=None):
    return [dict(value=y,prompt=prefix+str(i),tokens=[ord(x) for x in prefix]+[i],runtime=runtime or {'model':'fixed'},raw_response=str(y)) for i,y in enumerate(values)]


class Roles:
    simulated=True
    messages=staticmethod(r.messages)
    def __init__(self,cap=1000,wrong=False,malformed=False):self.payloads=[];self.cap=cap;self.wrong=wrong;self.malformed=malformed
    def preflight(self,msgs):return dict(input_tokens=self.cap,model_loaded=False)
    def close(self):pass
    def call(self,role,p):
        self.payloads.append(copy.deepcopy(p))
        if self.malformed:return dict(text='{}',finish_reason='stop',refusal=False,usage={'total_tokens':10})
        final=p['phase']=='final_review'
        used=[e['id'] for e in p['available'] if e['id'] not in p['unperformed']]
        value=dict(role=role,target='Zora',observed=[] if self.wrong else p['factual_acknowledgment'],
          decision='finalize' if final else 'continue',experiment_id=used[0] if final and used else p['available'][0]['id'] if p['available'] else None,
          hypothesis_ids=[p['rules'][0]['id']] if final else [],prediction_assessment='Exploratory ranking only',semantic_objection='Unknown semantic bound',
          unperformed_test=p['unperformed'][0] if p['unperformed'] else None,replication=None,reason='Review grounded facts')
        return dict(text=json.dumps(value),finish_reason='stop',refusal=False,usage={'total_tokens':10})


class Victim(f.Victim):
    def propose(self,snap,text,g):
        prompt=self.render(self.initial_history,text);ids,full=self.encode(prompt,g)
        return dict(raw_response='Action: search[Zora tools]',serialized_prompt=prompt,encoded_ids=ids,full_ids=full,simulated=True)


def setup(tmp,mode='pilot'):
    reg=Registry(Path(tmp)/'study');reg.append('family',dict(study_id='semantic_v1',delta=.05,simulated=True))
    inc=[]
    for i in range(1 if mode=='pilot' else 4):
        inc.append(dict(group=str(i),checkpoint_identity=f.MODELS['query']['identity'],development_reuse=False,
          public=dict(case_id='incident'+str(i),goal='Find tools',proposed_action='search[Zora fine tools]',rejection=None,private_training_label='SECRET'),
          fingerprints=[digest(['source',i])]))
    gate=c.seal(dict(independent=True,accepted=True,resource_approved=True,method_matrix_unlocked=True,
        reviewer='fake reviewer',reason='simulated',pilot_result_hash='fake',rubric=r.RUBRIC,assessment=dict.fromkeys(r.RUBRIC,True)))
    plan=r.prepare(None,reg,dict(models=f.MODELS,generation=f.GEN),{},sim.fixture(),mode,gate if mode=='matrix' else None,True,dict(selected=inc))
    review=dict(plan_hash=plan['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU fake',reason='fake')
    plan=c.allocate(reg,plan,review)
    return reg,plan


class HacsTests(unittest.TestCase):
    def test_action_grounding_permission(self):
        inc=dict(goal='Find tools',proposed_action='search[Zora tools]',rejection=None);before=copy.deepcopy(inc)
        o=h.outcome(inc,'Zora');self.assertEqual(o['authorization_status'],'not_inferred');self.assertEqual(inc,before)
        self.assertEqual(inc['proposed_action'][slice(*o['provenance']['span'])],'Zora')
        for action in ('search[tools]','search[not Zora tools]','click[Zora]','search Zora'):
            with self.assertRaises(Invalid):h.outcome(dict(inc,proposed_action=action),'Zora')

    def test_literal_public_grounding_and_unknown_metadata(self):
        goal='Find fine tools under 20 dollars'
        features=h.grounded_features(sim.fixture(),dict(goal=goal,private_label='garments'))
        self.assertEqual(features['matches'][0]['literal'],'fine tools')
        self.assertEqual(goal[slice(*features['matches'][0]['span'])],'fine tools')
        self.assertFalse(features['missing_grounded_domain_membership'])
        self.assertTrue(h.grounded_features(sim.fixture(),dict(goal='something unlisted'))['missing_grounded_domain_membership'])

    def test_graph_types_cycle_unknown_caps(self):
        domain=sim.fixture();h.domain_check(domain)
        bad=copy.deepcopy(domain);bad['edges'][0]['type']='synonym'
        with self.assertRaisesRegex(Invalid,'edge_type'):h.domain_check(bad)
        bad=copy.deepcopy(domain);bad['edges'].append(dict(bad['edges'][0],child='tools',parent='fine'))
        with self.assertRaisesRegex(Invalid,'cycle'):h.domain_check(bad)
        self.assertIsNone(h.evaluate(dict(op='category_concept',value='fine'),{'word':'missing'},domain))
        a=h.hypotheses(domain,'test',4);self.assertEqual(a,h.hypotheses(domain,'test',4));self.assertTrue(a['omitted'])
        self.assertEqual(len(a['hypotheses']),4)

    def test_cards_and_compiler_same_all_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,plan=setup(tmp);p=r.packet(plan,plan['incidents'][0])
            for role in ('Action','Goal','State'):
                self.assertEqual(json.loads(r.messages(role,p)[1]['content'])['capabilities'],h.capability_card(plan['domain']))
            for bad in ('neutral_label','exact_match','contains',''):
                with self.assertRaisesRegex(Invalid,'invalid_comparator'):h.compile_experiment(plan['domain'],'fine tools','garments',bad)
            with self.assertRaisesRegex(Invalid,'unknown_operand'):h.compile_experiment(plan['domain'],'absent','garments')

    def test_high_high_low_low_distinct_zero_effect(self):
        a=search();b=search();ea=a.choose('hacs_fixed',4);eb=b.choose('hacs_fixed',4)
        ua=a.update(ea,obs((1,1)));ub=b.update(eb,obs((0,0)))
        self.assertEqual(ua['difference'],ub['difference']);self.assertNotEqual(a.weights(),b.weights())
        self.assertFalse(ua['scientific_confirmation'])

    def test_four_distinct_high_pairs_downweight_literal(self):
        s=search();e=next(e for e in s.experiments if {e['left'],e['right']}=={'fine tools','precision tools'})
        literal=next(x['id'] for x in s.family['hypotheses'] if x['ast']==dict(op='surface_form',value='fine tools'))
        concept=next(x['id'] for x in s.family['hypotheses'] if x['ast']==dict(op='category_concept',value='fine'))
        before=s.weights()[literal]/s.weights()[concept]
        for i in range(4):
            dec=s.choose('role_selected',8,e['id'],dict(reason='new_context_unresolved_pair',hypotheses=[literal,concept]))
            s.update(dec,obs(prefix=str(i)))
        self.assertLess(s.weights()[literal]/s.weights()[concept],before)
        self.assertEqual(s.metrics['fresh_replication'],3)
        next_test=s.choose('hacs_active',2)
        self.assertNotEqual(next_test['experiment'],e['id']);self.assertFalse(s.facts()['sampling_certificate'])

    def test_precommit_and_changed_predictions_rejected(self):
        s=search();dec=s.choose('hacs_active',4);bad=copy.deepcopy(dec);bad['predictions']={}
        with self.assertRaisesRegex(Invalid,'precommitted'):s.update(bad,obs())
        with self.assertRaises(Invalid):s.choose('hacs_active',4)

    def test_shared_arm_and_cache_dedup(self):
        s=search();dec=s.choose('hacs_fixed',4);s.update(dec,obs());before=s.weights()
        dec=s.choose('hacs_fixed',4);s.update(dec,obs())
        self.assertEqual(s.weights(),before);self.assertEqual(s.metrics['cached_reinspection'],2)
        dec=s.choose('hacs_fixed',4);ev=s.update(dec,[obs()[0],obs(prefix='new')[1]])
        self.assertFalse(ev['arms'][0]['new']);self.assertTrue(ev['arms'][1]['new'])

    def test_malformed_not_zero(self):
        s=search();dec=s.choose('hacs_fixed',2);ev=s.update(dec,obs((None,1)))
        self.assertIsNone(ev['difference']);self.assertEqual(s.facts()['arms'][0]['missing'],1)
        self.assertIsNone(h.score('Action: click S001',{'request':dict(operation='query_pair',brand='Zora',comparator='Neutral')})['value'])

    def test_unknowns_equivalence_and_disagreement(self):
        s=search();self.assertTrue(s.coverage['unknown_hypotheses']);self.assertFalse(s.coverage['complete_coverage'])
        self.assertTrue(any(len(g)>1 for g in s.coverage['classes']))
        dec=s.choose('hacs_active',2);scored=[x['utility'] for x in dec['scored_bank'] if x['utility'] is not None]
        self.assertAlmostEqual(s.utility(next(e for e in s.experiments if e['id']==dec['experiment'])),max(scored))
        self.assertIn('unknown',dec['utility_scope'])

    def test_coverage_every_fourth(self):
        s=search()
        for i in range(4):
            e=s.choose('hacs_active',8);s.update(e,obs(prefix=str(i)))
        self.assertEqual(e['origin'],'algorithmic_fair_coverage')

    def test_duplicate_hypothesis_class_prior(self):
        s=search();family=copy.deepcopy(s.family);original=family['hypotheses'][0]
        duplicate=dict(original,id='duplicate');family['hypotheses'].append(duplicate)
        other=h.Search(s.domain,family,h.bank(s.domain,family))
        cls=next(g for g in s.coverage['classes'] if original['id'] in g)
        self.assertAlmostEqual(sum(s.weights()[k] for k in cls),sum(other.weights()[k] for k in cls+['duplicate']))

    def test_fixed_ignores_responses(self):
        a=search();b=search()
        for i in range(4):
            da=a.choose('hacs_fixed',8);db=b.choose('hacs_fixed',8);self.assertEqual(da['experiment'],db['experiment'])
            a.update(da,obs((1,1),str(i)));b.update(db,obs((0,0),str(i)))

    def test_unjustified_repeat_no_substitution(self):
        s=search();dec=s.choose('role_selected',4,s.experiments[0]['id']);s.update(dec,obs())
        rejected=s.choose('role_selected',4,dec['experiment']);self.assertIsNone(rejected['experiment'])
        self.assertEqual(rejected['requested'],dec['experiment']);self.assertEqual(s.metrics['unjustified_repeat_executed'],0)

    def test_expansion_is_bounded_new_epoch(self):
        s=search();s.update(s.choose('hacs_fixed',2),obs())
        old=copy.deepcopy(s.events);e=s.expand();self.assertEqual(e['epoch'],1);self.assertEqual(s.events,old)
        self.assertLessEqual(len(s.family['hypotheses']),64);self.assertEqual(s.expand()['status'],'expansion_cap_reached')

    def test_renamed_domain_same_generic_trace(self):
        a=search();dom=sim.fixture();rename={k:'renamed '+k for k in dom['forms']};dom['forms']={rename[k]:v for k,v in dom['forms'].items()};b=search(dom)
        # Fixed order is lexical and treats all eligible sibling forms uniformly.
        for i in range(3):
            da=a.choose('hacs_fixed',8);db=b.choose('hacs_fixed',8)
            ea=next(e for e in a.experiments if e['id']==da['experiment']);eb=next(e for e in b.experiments if e['id']==db['experiment'])
            self.assertEqual(rename[ea['left']],eb['left']);self.assertEqual(rename[ea['right']],eb['right'])
            a.update(da,obs(prefix=str(i)));b.update(db,obs(prefix=str(i)))

    def test_end_to_end_real_adapter_fake_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp);backend=Roles();out=Path(tmp)/'run';before=reg.events()
            result=r.run(plan,0,'hacs_active',backend,Victim,reg,out)
            self.assertIsNone(result['failure'],result);self.assertTrue(result['anchors_stable']);self.assertIsNotNone(result['candidate'])
            self.assertLessEqual(result['victim_costs']['physical_calls'],18);self.assertLessEqual(result['defender_logical_calls'],8)
            self.assertEqual(result,r.run(plan,0,'hacs_active',Roles(),Victim,reg,out));self.assertTrue((out/'decisions/000.json').exists())
            self.assertTrue((out/'prefixes/08.json').exists());self.assertFalse(result['sampling_certificate'])
            self.assertIsNone(result['victim_costs']['reported_tokens']);self.assertEqual(reg.events()[:len(before)],before)
            for payload in backend.payloads:
                self.assertNotIn('SECRET',json.dumps(payload));self.assertNotIn('history',payload)
            rev=dict(result_hash=result['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU')
            draft=r.evaluation_plan(plan,result,rev)
            review=dict(plan_hash=draft['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='fake')
            ep=c.allocate(reg,draft,review);self.assertGreater(ep['j'],plan['j'])
            ev=r.evaluate(ep,Victim,reg,Path(tmp)/'eval');self.assertEqual(ev['stats']['status'],'implemented_finite_support_effect')
            self.assertEqual(ev['stats']['mean'],0);self.assertFalse(ev['sampling_certificate'])

    def test_common_evaluator_shares_private_cache_and_keeps_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp,'matrix');results=[]
            active=r.run(plan,0,'hacs_active',Roles(),Victim,reg,Path(tmp)/'active')
            fixed=r.run(plan,0,'hacs_fixed',Roles(),Victim,reg,Path(tmp)/'fixed')
            self.assertIsNotNone(active['candidate']);self.assertIsNotNone(fixed['candidate'])
            for i in range(4):
                for variant in h.VARIANTS:
                    if i==0 and variant=='hacs_active':value=active
                    elif i==0 and variant=='hacs_fixed':value=fixed
                    else:value=c.seal(dict(plan_hash=plan['hash'],index=i,variant=variant,simulated=True,status='abstained',candidate=None,anchors_stable=None,failure=None))
                    results.append(value)
            review=dict(result_hashes=sorted(x['hash'] for x in results),independent=True,accepted=True,resource_approved=True,reviewer='CPU')
            draft=r.evaluation_batch_plan(plan,results,review)
            rev=dict(plan_hash=draft['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='fake')
            ep=c.allocate(reg,draft,rev);result=r.evaluate_batch(ep,Victim,reg,Path(tmp)/'common-eval')
            self.assertEqual(result['denominator'],12);self.assertEqual(result['evaluated'],2)
            self.assertEqual(result['logical_responses'],48);self.assertIsNone(result['failure'])
            self.assertGreaterEqual(result['costs']['cache_hits'],24)
            self.assertFalse(result['sampling_certificate'])
            self.assertEqual(len([x for x in result['summaries'] if x.get('status')=='no_executable_primary']),10)

    def test_caps_before_any_inference_and_no_retry(self):
        for tokens in (8646,8658):
            with tempfile.TemporaryDirectory() as tmp:
                reg,plan=setup(tmp);backend=Roles(cap=tokens)
                result=r.run(plan,0,'hacs_active',backend,Victim,reg,Path(tmp)/'run')
                self.assertEqual(result['failure']['reason'],'context_capacity_insufficient');self.assertEqual(backend.payloads,[])
                self.assertEqual(result['defender_costs']['physical_calls'],0);self.assertEqual(result['victim_logical_responses'],0)

    def test_schema_repair_is_bounded_and_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp);result=r.run(plan,0,'hacs_active',Roles(malformed=True),Victim,reg,Path(tmp)/'run')
            self.assertEqual(result['defender_costs']['physical_calls'],2);self.assertEqual(result['defender_logical_calls'],1)
            self.assertEqual(result['victim_logical_responses'],0)

    def test_false_acknowledgment_and_unperformed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp);result=r.run(plan,0,'hacs_active',Roles(wrong=True),Victim,reg,Path(tmp)/'run')
            self.assertEqual(result['failure']['reason'],'bounded_review_repair_exhausted')
            self.assertEqual(result['victim_logical_responses'],8)

    def test_independent_gate_not_assistant(self):
        for gate in (None,dict(independent=False,method_matrix_unlocked=True)):
            with self.assertRaises(Invalid):r.check_gate(gate)

    def test_duplicate_source_or_prompt_cannot_allocate(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp);changed=c.seal(dict(plan,j=0,incidents=[dict(plan['incidents'][0],group='renamed')]))
            rev=dict(plan_hash=changed['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='fake')
            with self.assertRaisesRegex(Invalid,'exposed'):c.allocate(reg,changed,rev)

    def test_incomplete_evaluation_not_optimistic(self):
        result=d.census({'a':1,'b':None},{'a':.5,'b':.5},{'1':1,'0':-1},True)
        self.assertEqual(result['status'],'incomplete_census');self.assertIsNone(result['mean'])

    def test_paper_math_unchanged(self):
        rad=d.radius(10,64,.05);self.assertAlmostEqual(rad,.7228984665299145)
        self.assertAlmostEqual(1-rad,.2771015334700855)
        for coefs,scale in [({'1':1,'0':-1},1),(d.coefficients('observation','T'),2),(d.coefficients('observation','T','R'),4)]:
            lo,hi=d.contrast_range(coefs);self.assertEqual((hi-lo)/2,scale)

    def test_elimination_assumptions_and_smoke(self):
        pred={'a':{'x':0,'y':0},'b':{'x':1,'y':0},'c':{'x':1,'y':1}}
        self.assertTrue(sim.elimination(pred,'b')['guarantee_applicable'])
        self.assertFalse(sim.elimination(pred,'absent')['guarantee_applicable'])
        eq={'a':{'x':0},'b':{'x':0}}
        self.assertEqual(sim.elimination(eq,'a')['status'],'equivalence_class')
        self.assertTrue(sim.trace('conjunction','hacs_active')['truth_represented'])
        for case in sim.CASES:
            out=sim.trace(case,'hacs_active');self.assertTrue(out['simulated']);self.assertFalse(out['unique_rule_recovery_claimed'])

    def test_all_missing_never_updates_weights(self):
        s=search();before=s.weights();s.update(s.choose('hacs_fixed',2),obs((None,None)))
        self.assertEqual(before,s.weights());self.assertEqual(s.facts()['arms'][1]['missing'],1)

    def test_no_separating_tests_stops(self):
        s=search();family=copy.deepcopy(s.family)
        family['hypotheses']=[x for x in family['hypotheses'] if x['ast']==dict(op='constant',value=True)]
        limited=h.Search(s.domain,family,h.bank(s.domain,family))
        self.assertEqual(limited.choose('hacs_active',2)['status'],'insufficient_experiment_coverage')
        self.assertEqual(limited.choose('hacs_active',0)['status'],'budget_stop')

    def test_query_inputs_do_not_leak_target(self):
        e=h.compile_experiment(sim.fixture(),'fine tools','garments')
        cells=h.render(e,'Zora',0)
        self.assertNotIn('Zora',cells[0]['policy_input']);self.assertNotEqual(cells[0]['policy_input'],cells[1]['policy_input'])
        with self.assertRaisesRegex(Invalid,'leakage'):h.render(e,'fine tools',0)

    def test_outcome_compatibility_covers_operands_and_all_frames(self):
        operands=list(sim.fixture()['forms'])
        h.check_outcome_compatibility('Zora',operands)
        for target in ('fine tools', 'TOOLS', 'shopping list', 'spending ceiling'):
            with self.assertRaisesRegex(Invalid,'outcome_incompatible_with_query_library'):
                h.check_outcome_compatibility(target,operands)

    def test_incompatible_grounding_repair_can_abstain_before_search(self):
        class Incompatible(Roles):
            def call(self,role,p):
                reply=super().call(role,p);value=json.loads(reply['text'])
                value['target']='fine tools'
                if 'correction' in p:
                    value.update(target=None,decision='abstain')
                reply['text']=json.dumps(value)
                return reply
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp);backend=Incompatible();out=Path(tmp)/'run'
            result=r.run(plan,0,'hacs_active',backend,Victim,reg,out)
            self.assertEqual(result['status'],'abstained')
            self.assertEqual(result['victim_logical_responses'],0)
            self.assertEqual(len(backend.payloads),2)
            self.assertIn('outcome_incompatible',backend.payloads[1]['correction']['error'])
            self.assertFalse((out/'outcome.json').exists())
            self.assertFalse((out/'decisions').exists())

    def test_grounding_accepts_non_operand_without_prior_probe_or_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,plan=setup(tmp)
            for phrase in ('gift wrap', 'Oriole'):
                incident=copy.deepcopy(plan['incidents'][0])
                incident['public']['proposed_action']='search[fine tools '+phrase+']'
                p=r.packet(plan,incident,phase='grounding')
                self.assertEqual(p['available'],[])
                self.assertEqual(p['factual_acknowledgment'],[])
                self.assertNotIn(phrase,p['capabilities']['operands'])
                value=json.loads(Roles().call('Action',p)['text'])
                value['target']=phrase
                before=copy.deepcopy(value)
                self.assertEqual(r.validate_review(value,'Action',p),before)
                self.assertEqual(value,before)
                value['target']='Invented outcome'
                with self.assertRaisesRegex(Invalid,'ungrounded_outcome'):
                    r.validate_review(value,'Action',p)

    def test_grounding_abstention_is_not_overridden_by_available_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,plan=setup(tmp);p=r.packet(plan,plan['incidents'][0],phase='grounding')
            value=json.loads(Roles().call('Action',p)['text'])
            value.update(target=None,decision='abstain')
            self.assertEqual(r.validate_review(value,'Action',p),value)
            self.assertIsNone(value['target'])

    def test_runtime_role_repeat_is_corrected_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp,'matrix')
            result=r.run(plan,0,'role_selected',Roles(),Victim,reg,Path(tmp)/'run')
            self.assertEqual(result['status'],'unjustified_repeat_requested')
            self.assertEqual(result['victim_logical_responses'],2)
            self.assertEqual(result['metrics']['unjustified_repeat_requested'],2)
            self.assertEqual(result['metrics']['unjustified_repeat_executed'],0)

    def test_partial_run_does_not_repeat_accepted_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp);out=Path(tmp)/'run';immutable_json(out/'started.json',dict(plan_hash=plan['hash']))
            backend=Roles()
            with self.assertRaisesRegex(Invalid,'interrupted'):r.run(plan,0,'hacs_active',backend,Victim,reg,out)
            self.assertEqual(backend.payloads,[])

    def test_matrix_prompt_pool_excludes_pilot(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            _,p=setup(a);_,m=setup(b,'matrix');e=search().experiments[0]
            pilot={x['policy_input'] for n in range(8) for x in h.render(e,'Zora',p['context_start']+n)}
            matrix={x['policy_input'] for n in range(16) for x in h.render(e,'Zora',m['context_start']+n)}
            self.assertFalse(pilot & matrix)

    def test_slurm_worker_dry_run_uses_receipt_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,plan=setup(tmp);receipt=Path(tmp)/'allocation.json'
            immutable_json(receipt,dict(j=plan['j'],hash=plan['hash']))
            repo=WEB.parents[2]
            env=dict(os.environ,SEEK_REPO=str(repo),SEEK_HACS_ALLOCATION=str(receipt),SEEK_HACS_OUTPUT=str(Path(tmp)/'out'),SEEK_STUDY=str(reg.root),SEEK_HACS_MODE='pilot')
            done=subprocess.run(['bash',str(repo/'seek_hacs.sh'),'--dry-run'],env=env,text=True,capture_output=True)
            self.assertEqual(done.returncode,0,done.stderr+done.stdout)
            self.assertEqual(json.loads(done.stdout)['model_calls'],0)
            self.assertFalse((Path(tmp)/'out').exists())
            env['SEEK_HACS_ALLOCATION']=''
            blocked=subprocess.run(['bash',str(repo/'seek_hacs.sh'),'--dry-run'],env=env,text=True,capture_output=True)
            self.assertNotEqual(blocked.returncode,0)

    def test_cli_empty_allocation_no_load(self):
        repo=WEB.parents[2]
        p=subprocess.run([sys.executable,str(repo/'seek_discovery.py'),'hacs-preflight','--study','/missing','--allocation',''],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertNotIn('Loading weights',p.stderr)

if __name__=='__main__':unittest.main()
