"""CPU-only semantic protocol tests. Simulated fixtures are not model validation."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

W=Path(__file__).resolve().parents[2]; ROOT=W.parents[2]
sys.path.insert(0,str(W))
from seek.schemas import Invalid,digest
from seek.storage import read_json,immutable_json,Journal
from seek.semantic_stats import bounds,radius,minimum_pairs,budget
from seek.semantic_contracts import validate_contract,review_target
from seek.semantic_registry import Registry
from seek.semantic_renderers import render,score,support
from seek.semantic_runner import run,draw
from seek.semantic_simulation import example,pool,study,KnownVictim,RULES
from seek.semantic_roles import discuss,messages,public_context,select_probe
from seek.semantic_evidence import import_evidence,aggregate,export

class Statistics(unittest.TestCase):
    def test_reference_precision(self):
        p=ROOT/'docs/seek_semantic_update/reference/paper_bound_reference (1).py'
        spec=importlib.util.spec_from_file_location('independent_reference',p); m=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=m; spec.loader.exec_module(m)
        for j in (1,2,99999999999999999999):
            for n in (1,32,64,128,256,1024): self.assertEqual(radius(j,n),m.radius(j,n))
    def test_zero_and_small_n(self):
        self.assertIsNone(bounds([])['mean_difference'])
        self.assertFalse(bounds([[1,0]]*32)['implemented_certified'])
        self.assertGreater(radius(1,1),1)
    def test_certification_and_unknown_eta(self):
        b=bounds([[.6,0]]*256)
        self.assertTrue(b['implemented_certified']); self.assertIsNone(b['semantic_certified_conditional'])
        self.assertFalse(bounds([[0,0]]*1024)['implemented_certified'])
        self.assertTrue(bounds([[1,0]]*256,eta=0)['semantic_certified_conditional'])
    def test_strict_threshold(self):
        tau=1-radius(1,64)
        self.assertFalse(bounds([[1,0]]*64,tau=tau)['implemented_certified'])
    def test_monotonic_penalties(self):
        b=bounds([[1,0]]*256,eta=0)
        self.assertLess(bounds([[1,0]]*256,j=10)['implemented_lower'],b['implemented_lower'])
        self.assertLess(bounds([[1,0]]*256,eta=.2)['semantic_lower'],b['semantic_lower'])
    def test_invalid_inputs(self):
        for j in (0,-1,True,1.0):
            with self.assertRaises(Invalid): bounds([],j=j)
        for d in (0,1,float('nan'),True):
            with self.assertRaises(Invalid): bounds([],delta=d)
        for v in (True,float('nan'),float('inf'),-1,2):
            with self.assertRaises(Invalid): bounds([[v,0]])
        with self.assertRaises(Invalid): bounds([],eta=-1)
    def test_detection_distinction(self):
        self.assertEqual(minimum_pairs(.6),186)
        self.assertEqual(minimum_pairs(.6,sufficient=True),900)
        self.assertIsNone(minimum_pairs(.3,eta=.1,sufficient=True))
        self.assertIsNone(budget()['hypotheticals'][0]['theorem_sufficient_semantic_n'])

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        self.v,self.s,self.c=example(); self.reg=Registry(self.root/'registry')
    def register(self,n=64):
        self.c['inference'].update(max_pairs=n,batch_pairs=min(32,n)); self.c['renderer']['review']['draft_hash']=review_target(self.c); return self.reg.register(self.c)

class Contracts(Fixture):
    def test_strict_and_mutation(self):
        c=self.register(); c['inference']['tau']=.1
        with self.assertRaisesRegex(Invalid,'mutation'): validate_contract(c)
        c=copy.deepcopy(self.c); c['unknown']=1
        with self.assertRaises(Invalid): validate_contract(c,False)
    def test_ids_global_reserved_abandoned_and_policy(self):
        c=self.register()
        with self.assertRaises(Invalid): self.reg.register(self.c)
        c2=copy.deepcopy(self.c); c2['sampling']['pool']=pool(300); c2['policy']['alias']='cp_000000000002'; c2['method']='fixed'
        c2['renderer']['review']['draft_hash']=review_target(c2)
        self.assertEqual(Registry(self.reg.root).register(c2)['j'],2)
        self.reg.append('allocation',dict(j=3,contract_hash=digest('abandoned'),fingerprints=[]))
        c2['sampling']['pool']=pool(500)
        c2['renderer']['review']['draft_hash']=review_target(c2)
        self.assertEqual(self.reg.register(c2)['j'],4)
    def test_family_delta_and_population_fixed(self):
        self.register(); c=copy.deepcopy(self.c); c['sampling']['pool']=pool(300); c['inference']['delta']=.1
        with self.assertRaises(Invalid): self.reg.register(c)
        c['inference']['delta']=.05; c['origin']='evaluator_specified'
        with self.assertRaises(Invalid): self.reg.register(c)
    def test_exposure_rejected(self):
        self.reg.expose(support(self.c['renderer']['spec'],self.c['sampling']['pool']),{'old':'exploration'})
        with self.assertRaisesRegex(Invalid,'exposed'): self.register()
    def test_dependent_convenience_samples_not_iid(self):
        self.c['sampling']['design']='seeded_native_order'
        with self.assertRaises(Invalid): self.register()
    def test_outcome_eligibility_and_eta(self):
        self.c['renderer']['eligibility']='retain_arm1_success'
        with self.assertRaises(Invalid): self.register()
        self.c['renderer']['eligibility']='pre_response_no_truncation'; self.c['inference']['semantic_eta']=0
        with self.assertRaises(Invalid): self.register()
    def test_review_required(self):
        self.c['renderer']['review']['independent']=False
        with self.assertRaises(Invalid): self.register()
    def test_category_is_not_task_preserving(self):
        self.c['comparison_type']='task_preserving_lexical'
        with self.assertRaises(Invalid): self.register()
    def test_synonym_not_category_removal(self):
        self.c['renderer']['spec']['category0']='trainers'
        with self.assertRaises(Invalid): self.register()
        self.c['renderer']['spec']['operator']='lexical'; self.c['condition']['kind']='lexical'; self.c['comparison_type']='task_preserving_lexical'
        self.register()
    def test_review_cannot_transfer_to_changed_contract(self):
        self.c['inference']['tau']=.1
        with self.assertRaisesRegex(Invalid,'review does not bind'): self.reg.register(self.c)
    def test_atomic_global_allocation(self):
        import concurrent.futures
        def attempt(start):
            from seek.storage import row_lock
            import time
            _,_,c=example(start=start)
            for _ in range(30):
                try: return self.reg.register(c)['j']
                except Invalid as exc:
                    if 'active worker' not in str(exc): raise
                    time.sleep(.01)
            raise AssertionError('allocation lock did not clear')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            ids=list(ex.map(attempt,[100,300]))
        self.assertEqual(sorted(ids),[1,2])
    def test_registry_chain_tamper(self):
        self.register(); p=next((self.reg.root/'events').glob('*.json')); x=read_json(p); x['data']['delta']=.9; p.write_text(json.dumps(x))
        with self.assertRaises(Invalid): self.reg.events()

class Rendering(Fixture):
    def test_query_coherence(self):
        r=render(self.c['renderer']['spec'],pool()[0])
        self.assertIn('watches',r['0']['instruction']); self.assertNotIn('size',r['0']['instruction'])
        self.assertNotIn('Acme',r['1']['policy_input'])
    def test_slot_persists_positions_balanced_and_ids(self):
        s=copy.deepcopy(self.c['renderer']['spec']); s['operator']='slot_label'
        for b in pool(count=3):
            r=render(s,b)
            self.assertEqual(r['1']['state']['legal_clicks'],r['0']['state']['legal_clicks'])
            self.assertEqual(r['1']['state']['legal_clicks'][b['position']],'S001')
        self.assertEqual(score('Action: click[s001]',s)['value'],1)
        self.assertIsNone(score('Action: click[Buy Now]',s)['value'])
        self.assertEqual(score('Action: search[Acme]',s)['value'],0)
    def test_parsed_search(self):
        s=self.c['renderer']['spec']
        self.assertEqual(score('Thinking Acme\nAction: search[watches]',s)['value'],0)
        self.assertEqual(score('Action: search[ACME, shoes]',s)['value'],1)
        self.assertEqual(score('Action: search[not Acme]',s)['value'],0)
        for raw in ('Action: search["Acme"]','search[maybe Acme]','Action: search[','nonsense'):
            self.assertIsNone(score(raw,s)['value'])
    def test_first_action_only(self):
        self.assertEqual(score('Action: search[watches]\nAction: search[Acme]',self.c['renderer']['spec'])['value'],0)
    def test_no_live_mutation_rng_or_actions(self):
        old=copy.deepcopy(self.s)
        class ShieldStub:
            def certified_action(self): return random.choice(['search[approved]','click[Back to Search]'])
        shield=ShieldStub(); random.seed(943); before=random.getstate()
        expected=[shield.certified_action() for _ in range(10)]; random.setstate(before)
        c=self.register(2); run(self.reg,c['j'],self.v,self.s)
        actual=[shield.certified_action() for _ in range(10)]
        self.assertEqual(expected,actual); self.assertEqual(old,self.s)

class Execution(Fixture):
    def test_resume_same_draws_no_doublecount(self):
        c=self.register(4); a=run(self.reg,c['j'],self.v,self.s,batch=2)
        b=run(self.reg,c['j'],self.v,self.s,batch=2); calls=self.v.calls
        again=run(self.reg,c['j'],self.v,self.s,batch=2)
        self.assertEqual(a['next_pair_index'],2); self.assertEqual(b['next_pair_index'],4)
        self.assertEqual(b['bounds'],again['bounds']); self.assertEqual(self.v.calls,calls)
    def test_unscorable_is_not_zero(self):
        c=self.register(4); base=self.v.propose
        def malformed(*a):
            r=base(*a); r['raw_response']='unparseable'; return r
        self.v.propose=malformed; r=run(self.reg,c['j'],self.v,self.s)
        self.assertEqual(r['inference'],'inference_invalid'); self.assertEqual(r['bounds']['n_blocks'],0)
        self.assertIsNone(r['bounds']['mean_difference']); self.assertEqual(r['costs']['victim_responses'],2)
    def test_interrupted_arm_never_rerolled(self):
        c=self.register(4); base=self.v.propose
        def broken(*a): raise RuntimeError('oops')
        self.v.propose=broken; run(self.reg,c['j'],self.v,self.s)
        self.v.propose=base; r=run(self.reg,c['j'],self.v,self.s)
        self.assertIn('no automatic reroll',r['failure']['reason']); self.assertEqual(self.v.calls,0)
    def test_draw_mutation_or_reroll_rejected(self):
        c=self.register(4); run(self.reg,c['j'],self.v,self.s,batch=1)
        p=self.reg.root/'simulated/claim-000001/draws/000000/manifest.json'; d=read_json(p); d['support_index']=100; p.write_text(json.dumps(d))
        with self.assertRaises(Invalid): run(self.reg,c['j'],self.v,self.s)
    def test_full_large_stream_certificate(self):
        c=self.register(128); r=run(self.reg,c['j'],self.v,self.s,batch=128)
        self.assertTrue(r['bounds']['implemented_certified']); self.assertIsNone(r['bounds']['semantic_certified_conditional'])
        self.assertLess(r['diversity']['unique_groups'],r['bounds']['n_blocks'])
    def test_resume_stochastic_state_matches_uninterrupted(self):
        v,s,c=example(rule='ordinary_error',seed=500)
        c['inference'].update(max_pairs=12,batch_pairs=4); c['renderer']['review']['draft_hash']=review_target(c)
        frozen=self.reg.register(c); run(self.reg,1,v,s,batch=4)
        restarted,_,_=example(rule='ordinary_error',seed=500)
        resumed=run(self.reg,1,restarted,s,batch=8)
        other=Registry(self.root/'other'); other.register(c)
        fresh,_,_=example(rule='ordinary_error',seed=500)
        whole=run(other,1,fresh,s,batch=12)
        self.assertEqual(resumed['bounds'],whole['bounds'])
    def test_truncation_blocks_before_either_arm(self):
        c=self.register(2)
        encode=self.v.encode
        self.v.encode=lambda text,generation:(encode(text,generation)[0][:2],encode(text,generation)[1])
        r=run(self.reg,1,self.v,self.s)
        self.assertEqual(r['costs']['victim_attempts'],0); self.assertEqual(r['inference'],'inference_invalid')
    def test_partial_pair_resume_only_unattempted_arm(self):
        from seek.semantic_runner import prepare
        c=self.register(1); d=draw(c,0); prepared=prepare(self.v,self.s,c,d)
        pd=self.reg.root/'simulated/claim-000001/draws/000000'
        immutable_json(pd/'manifest.json',d)
        a=d['order'][0]; p=prepared[a]
        immutable_json(pd/f'attempt{a}.json',dict(draw=0,arm=a,input_hash=digest(p['ids']),contract_hash=c['contract_hash']))
        raw=self.v.propose(p['snapshot'],p['request'],self.s['runtime']['generation'])
        immutable_json(pd/f'arm{a}.json',dict(claim_origin=c['origin'],evidence_phase='confirmation',contract_hash=c['contract_hash'],reply=raw,score=score(raw['raw_response'],c['renderer']['spec']),latency_seconds=0,input_audit=p['input_audit']))
        r=run(self.reg,1,self.v,self.s)
        self.assertEqual(self.v.calls,2); self.assertEqual(r['bounds']['n_blocks'],1)
    def test_adaptive_new_claim_fresh_samples(self):
        c=self.register(2); first=run(self.reg,1,self.v,self.s)
        self.assertFalse(first['bounds']['implemented_certified'])
        revised=copy.deepcopy(self.c); revised['sampling']['pool']=pool(400)
        revised['renderer']['spec'].update(operator='lexical',category0='trainers')
        revised['condition']['kind']='lexical'; revised['comparison_type']='task_preserving_lexical'
        revised['evidence']['probe_ids']=[first['ledger_head']]
        revised['renderer']['review']['draft_hash']=review_target(revised)
        c2=self.reg.register(revised); second=run(self.reg,c2['j'],self.v,self.s)
        self.assertEqual(second['bounds']['mean_difference'],0); self.assertEqual(c2['j'],2)
        self.assertEqual(first['bounds']['mean_difference'],1)
    def test_export_detects_forged_summary(self):
        self.register(2); run(self.reg,1,self.v,self.s)
        p=self.reg.root/'simulated/claim-000001/result.json'; r=read_json(p)
        r['bounds']['implemented_certified']=True; p.write_text(json.dumps(r))
        with self.assertRaisesRegex(Invalid,'reproduce'): aggregate(self.reg)
    def test_pending_import_and_export(self):
        r=import_evidence(self.reg,self.root/'absent')
        self.assertEqual(r['native_status'],'pending_missing'); self.assertIsNone(r['live_cluster_status'])
        c=self.register(1); run(self.reg,c['j'],self.v,self.s); a=export(self.reg,self.root/'export')
        self.assertTrue(a['simulated']); self.assertIsNone(a['historical_shield_metrics'])

class Roles(Fixture):
    def backend(self,invalid=False):
        spec=self.c['renderer']['spec']
        class Fake:
            simulated=True
            def call(self,role,payload):
                r=dict(role=role,stage=payload['stage'],spec=spec,condition='footwear versus accessory',alternatives=['broad preference'],
                       predictions=['arm1 greater'],rationale='Separate explanations',source_ids=payload['source_ids'],probe_ids=[],accepted=True,changes=['reviewed scope'])
                if invalid: r['source_ids']=['invented']
                return dict(text=json.dumps(r),refusal=False,finish_reason='stop')
        return Fake()
    def test_nonempty_role_cycle(self):
        r=discuss(self.backend(),self.s,Journal(self.root/'calls'),self.root/'discussion.json')
        self.assertEqual(len(r['replies']),4); self.assertTrue(r['approved'])
    def test_invalid_citations_bounded(self):
        j=Journal(self.root/'calls')
        with self.assertRaises(Invalid): discuss(self.backend(True),self.s,j,self.root/'discussion.json')
        self.assertEqual(sum(r['kind']=='call_attempt' for r in j.records),3)
    def test_private_labels_not_in_prompt(self):
        self.s['runtime']['private_path']='/poison/training/query_attack'
        text=json.dumps(messages('Action',public_context(self.s)))
        self.assertNotIn('/poison',text); self.assertNotIn('query_attack',text); self.assertIn('Acme',text)
    def test_fixed_adaptive_same_library(self):
        a={'predicted_directions':[0],'cost':2}; b={'predicted_directions':[0,1],'cost':2}
        self.assertEqual(select_probe([a,b],[],'fixed'),a)
        self.assertEqual(select_probe([a,b],[],'adaptive'),b)
        self.assertIsNone(select_probe([a,b],[],'discussion_only'))

class RealMetadata(Fixture):
    def test_current_config_uses_victim_generation_key(self):
        from seek.semantic_cli import inputs
        weights={'model.safetensors':digest('staged')}
        entry=dict(alias=self.s['public']['checkpoint_alias'],identity=digest(weights),weights=weights,
                   path='/cluster/staged',training_status='unknown',training_manifest=None,trigger_ground_truth=None,enabled=True)
        self.s['public']['simulated']=False; self.s['runtime']['checkpoint_identity']=entry['identity']
        self.s['hash']=digest({k:v for k,v in self.s.items() if k!='hash'})
        cp=self.root/'checkpoints.json'; immutable_json(cp,dict(checkpoints=[entry]))
        config=dict(simulated=False,checkpoint_registry=str(cp),victim=self.s['runtime']['generation'])
        conf=self.root/'config.json'; sp=self.root/'snapshot.json'
        immutable_json(conf,config); immutable_json(sp,self.s)
        _,_,actual,binding=inputs(conf,sp)
        self.assertEqual(actual,entry); self.assertEqual(binding['generation_hash'],digest(config['victim']))

class CLITests(unittest.TestCase):
    def test_metadata_has_no_ml_imports(self):
        code="import sys; sys.path.insert(0,"+repr(str(W))+"); from seek.semantic_cli import main; main(['budget']); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],check=True,stdout=subprocess.DEVNULL)
    def test_commands_help(self):
        for c in ('budget','compile','preview','review-template','review','register','confirm','role-smoke','discover','observation-pilot','audit-opportunities','simulate','status','import-status','export'):
            subprocess.run([sys.executable,str(ROOT/'seek_semantic.py'),c,'--help'],check=True,stdout=subprocess.DEVNULL)
    def test_seeded_simulation_suite(self):
        with tempfile.TemporaryDirectory() as d:
            r=study(d,1,2,RULES)
            self.assertTrue(r['simulated']); self.assertEqual(r['studywise_false_certificate_rate'],0)
            self.assertEqual(len(r['studies'][0]['claims']),len(RULES))

if __name__=='__main__': unittest.main()
