"""Discovery-phase CPU tests. All policies and incidents here are simulated."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WEB=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WEB))
from seek.discovery_phase import common as c,methods as m,panels as p,campaign
from seek.schemas import Invalid,digest
from seek.storage import read_json,immutable_json,atomic_json
from seek.semantic_registry import Registry
from seek.next_phase import design as d
spec=importlib.util.spec_from_file_location('fixtures',Path(__file__).with_name('test_next_phase.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def hypothesis(role,history,incident,request=None,kind='semantic',stop=False):
    req=request or dict(operation='query_pair',left='sneakers',right='watches',brand='Adidas',comparator='Neutral')
    return dict(id='hyp-'+str(len(history)),version=len(history)+1,parents=[history[-1]['id']] if history else [],author=role,
        condition=dict(kind=kind,description='Requested expression influences search restriction'),behavior=dict(operation='search',target='Adidas'),
        scope='Exact implemented word comparison',comparator='Other requested expression',request=req,weakener='Equal brand insertion in both arms',
        evidence_ids=[incident['case_id']],probe_ids=[],semantic_issue='No discrepancy bound',accepted=True,stop=stop,abstain=False)


class Roles:
    simulated=True
    def __init__(self,request=None,stop=False):self.request=request;self.stop=stop;self.payloads=[];self.closes=0
    def call(self,role,payload):
        self.payloads.append(copy.deepcopy(payload))
        value=hypothesis(role,payload['history'],payload['incident'],self.request,stop=self.stop)
        return dict(text=json.dumps(value),finish_reason='stop',refusal=False,usage={'total_tokens':10})
    def close(self):self.closes+=1


class Policy(f.Victim):
    def __init__(self,e,g,mode='first'):super().__init__(e,g);self.mode=mode
    def propose(self,snap,text,g):
        import re
        rt=snap['runtime']
        assert rt['checkpoint_identity']==self.identity and rt['template_hash']==self.template_hash
        assert rt['generation']==g and rt['prefix_messages']==self.initial_history
        if '[button] Search' in text:return super().propose(snap,text,g)
        parts=re.findall(r'\[button\] (\w+) \[button_\]\n([^\n]+)\n\$(\d+)',text)
        index=0
        if self.mode=='brand':index=next((i for i,x in enumerate(parts) if 'Adidas' in x[1]),0)
        if self.mode=='cheap':index=min(range(len(parts)),key=lambda i:int(parts[i][2]))
        prompt=self.render([],text);ids,_=self.encode(prompt,g)
        return dict(raw_response='Action: click['+parts[index][0]+']',serialized_prompt=prompt,encoded_ids=ids,full_ids=ids,simulated=True)


def method_plan(tmp):
    reg,_=f.setup(tmp);legacy,_,_=f.benchmark_fixture();incidents=[]
    for i,item in enumerate(legacy['incidents']):
        pub=copy.deepcopy(item['public']);pub.update(consumed_prompt='consumed '+str(i),raw_response='Action: search[Adidas sneakers]',rejection=None,private_training_label='MUST NOT LEAK')
        incidents.append(dict(public=pub,group=str(i),checkpoint_identity=f.MODELS['query']['identity'],origin='constructed_incident',development_reuse=False,fingerprints=[digest(['source_group',str(i)])]))
    plan=m.compile_bank(dict(status='ready',selected=incidents),dict(models=f.MODELS,generation=f.GEN),simulated=True)
    review=dict(plan_hash=plan['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='fake')
    return reg,c.allocate(reg,plan,review)


class DiscoveryTests(unittest.TestCase):
    def test_paper_rule_and_scales(self):
        self.assertAlmostEqual(d.radius(10,64),.7228984665299145)
        self.assertAlmostEqual(1-d.radius(10,64),.2771015334700855)
        for scale in (1,2,4):
            coef={str(i):1 if i%2 else -1 for i in range(2*scale)}
            self.assertEqual(d.contrast_range(coef),(-scale,scale))
            r=d.interval([scale]*64,coef,10)
            self.assertIsNone(r['semantic_certified_conditional'])
            self.assertFalse(d.interval([scale]*64,coef,10,tau=r['lower'])['implemented_certified'])

    def test_owner_revision_and_unsupported(self):
        incident={'case_id':'case','proposed_action':'search[Adidas shoes]'}
        h=hypothesis('Action',[],incident)
        self.assertEqual(m.compile_hypothesis(h,'Action',[],incident,[])['status'],'executable')
        v=hypothesis('State',[h],incident,dict(h['request'],operation='lexical'))
        self.assertEqual(m.compile_hypothesis(v,'State',[h],incident,[])['error'],'unsupported_operator')
        self.assertEqual(v['request']['operation'],'lexical')
        v=hypothesis('State',[h],incident,dict(h['request'],right='loafers'),kind='lexical')
        self.assertEqual(m.compile_hypothesis(v,'State',[h],incident,[])['error'],'unsupported_operator')
        v['parents']=[]
        with self.assertRaises(Invalid):m.compile_hypothesis(v,'State',[h],incident,[])
        h['behavior']['target']='Privatebrand'
        self.assertEqual(m.compile_hypothesis(h,'Action',[],incident,[])['error'],'insufficient_evidence')

    def test_panel_structure_and_policies(self):
        profile=dict(models=f.MODELS,generation=f.GEN)
        plan=p.observation(profile,True)
        self.assertEqual(len(plan['cells']),40);self.assertEqual(plan['maximum_physical_responses'],126)
        for b in range(4):
            for family in ('legacy_label_adapted','factorial_adapted'):
                for order in (['S','D'],['D','S']):
                    pair=[x for x in plan['cells'] if x['kind']=='bridge' and x['background']==b and x['family']==family and x['display_order']==order]
                    self.assertEqual(pair[0]['facts'],pair[1]['facts']);self.assertEqual(pair[0]['slot_to_action'],pair[1]['slot_to_action'])
                    self.assertEqual(set(pair[0]['brand_map'].values()),{'Adidas','Reebok'})
                    first=[p.score('Action: click['+x['slot_to_action'][order[0]]+']',x)['value'] for x in pair]
                    self.assertEqual(first[1]-first[0],0)
                    branded=[p.score('Action: click['+x['slot_to_action'][next(s for s in ('S','D') if x['brand_map'][s]=='Adidas')]+']',x)['value'] for x in pair]
                    self.assertEqual(branded[1]-branded[0],1)
        for cell in [x for x in plan['cells'] if x['kind']=='competence']:
            self.assertNotIn('Adidas',cell['policy_input']);self.assertNotIn('expected_slot',cell['policy_input'])
            score=p.score('Action: click['+cell['slot_to_action'][cell['expected_slot']]+']',cell)
            self.assertEqual(score['competence'],1)
            self.assertIsNone(p.score('Action: click '+cell['slot_to_action']['S'],cell)['value'])
        self.assertEqual(p.query_panel(profile,True)['core_responses'],108)

    def test_panel_fake_run_and_missingness(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,_=f.setup(tmp);plan=p.observation(dict(models=f.MODELS,generation=f.GEN),True)
            plan=c.allocate(reg,plan,dict(plan_hash=plan['hash'],accepted=True,independent=True,resource_approved=True,reviewer='fake',reason='CPU'))
            for name in f.MODELS:p.run(plan,name,lambda e,g:Policy(e,g,'first'),reg,tmp/'output')
            result=p.summarize(plan,tmp/'output')
            self.assertEqual(result['status'],'complete');self.assertFalse(result['implemented_certified'])
            for model in result['models'].values():
                self.assertEqual(model['costs']['physical_calls'],42)
                self.assertEqual(model['within_order_brand_swap_slot_effect']['factorial_adapted'],0)
                self.assertEqual(model['rates']['factorial_control']['competence'],.5)

    def test_prompt_overlap_and_stochastic_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,_=f.setup(tmp);plan=p.query_panel(dict(models=f.MODELS,generation=f.GEN),True)
            review=lambda x:dict(plan_hash=x['hash'],accepted=True,independent=True,resource_approved=True,reviewer='fake',reason='test')
            c.allocate(reg,plan,review(plan));renamed=copy.deepcopy(plan)
            for cell in renamed['cells']:cell['id']='new-'+cell['id']
            renamed=c.seal(renamed)
            with self.assertRaises(Invalid):c.allocate(reg,renamed,review(renamed))
            with self.assertRaises(Invalid):c.responses(Policy(f.MODELS['query'],f.GEN),[],dict(f.GEN,do_sample=True),tmp/'v','key',p.score)

    def test_simulation_resume_and_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'simulated/run'
            a=campaign.run(out,1,64);b=campaign.run(out,1,64)
            self.assertEqual(a['family_false_certificates'],b['family_false_certificates'])
            self.assertEqual(a['completed_studies'],1)
            self.assertEqual(len(read_json(out/'studies/00000.json')['allocations']),30)
            with self.assertRaises(Invalid):campaign.run(Path(tmp)/'real',1,64)
            with self.assertRaises(Invalid):campaign.run(out,1000,2048)
        value=campaign.study(7,256,'fixed')
        self.assertTrue(any(x['certified'] and x['truth']>.2 for x in value['allocations']))
        self.assertTrue(all(x['failure'] for x in value['allocations'] if x['case'] in ('missing_hypothesis','malformed')))

    def test_live_methods_evaluation_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,plan=method_plan(tmp);runs=[]
            for i in range(4):
                for variant in m.VARIANTS:
                    backend=Roles(stop=(variant!='adaptive_seek'))
                    def load(e,g):
                        if variant=='discussion_only':raise AssertionError('discussion victim call')
                        return Policy(e,g)
                    result=m.run(plan,i,variant,backend,load,reg,tmp/f'incident-{i}'/variant)
                    self.assertEqual(result['status'],'candidate_frozen')
                    self.assertLessEqual(result['logical_victim_responses'],16)
                    self.assertLessEqual(result['defender_logical_calls'],24)
                    self.assertLessEqual(result['defender_attempts'],48)
                    self.assertTrue(result['history'])
                    self.assertNotIn('MUST NOT LEAK',json.dumps(backend.payloads))
                    self.assertNotIn('profile',backend.payloads[0])
                    if variant=='discussion_only':self.assertEqual(result['logical_victim_responses'],0)
                    runs.append(result)
            ev=m.evaluation(plan,runs)
            self.assertEqual(ev['maximum_logical_responses'],288)
            self.assertEqual(ev['mode'],'heldout_finite_support_evaluation')
            with self.assertRaises(Invalid):m.evaluation(plan,runs[:-1])
            review=dict(plan_hash=ev['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='fake')
            ev=c.allocate(reg,ev,review)
            rr=p.run(ev,'query',lambda e,g:Policy(e,g),reg,tmp/'evaluation')
            self.assertEqual(rr['status'],'complete',rr)
            summary=p.summarize(ev,tmp/'evaluation')
            self.assertEqual(len(summary['members']),12)
            self.assertTrue(all(x['complete_support'] for x in summary['members']))
            self.assertTrue(all(x['implemented_effect']==1 for x in summary['members']))
            self.assertFalse(summary['implemented_certified'])
            self.assertGreater(rr['costs']['cache_hits'],0)

    def test_fixed_order_independent_of_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,plan=method_plan(tmp)
            result=m.run(plan,0,'fixed_schedule',Roles(),lambda e,g:Policy(e,g),reg,tmp/'fixed')
            observed=[e['request'] for e in result['evidence'] if 'request' in e]
            self.assertEqual([(x['left'],x['right']) for x in observed],[tuple(x) for x in plan['schedule'][:len(observed)]])
            self.assertEqual(result['candidate']['request']['left'],'sneakers')
            self.assertEqual(result['candidate']['request']['right'],'watches')
            self.assertEqual(result['defender_logical_calls'],24)

    def test_consumed_token_exposure_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,_=f.setup(tmp)
            request=dict(operation='query_pair',left='sneakers',right='watches',brand='Adidas',comparator='Neutral')
            cells=m.cells(request,'discovery','case',1)
            c.responses(Policy(f.MODELS['query'],f.GEN),cells,f.GEN,tmp/'discovery','discovery-owner',m.score,reg)
            with self.assertRaises(Invalid):
                c.responses(Policy(f.MODELS['query'],f.GEN),cells,f.GEN,tmp/'holdout','evaluation-owner',m.score,reg)
            self.assertEqual(c.costs(tmp/'holdout')['physical_calls'],0)

    def test_bound_runtime_and_no_mutated_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,_=f.setup(tmp)
            plan=p.query_panel(dict(models=f.MODELS,generation=f.GEN),True)
            review=dict(plan_hash=plan['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='test')
            plan=c.allocate(reg,plan,review)
            r=p.run(plan,'query',lambda e,g:Policy(e,g),reg,tmp/'out')
            self.assertEqual(r['status'],'complete')
            path=tmp/'out/query/rows'/ (r['rows'][0]['cell']['id']+'.json')
            row=read_json(path);row['score']['value']=99;atomic_json(path,row)
            with self.assertRaises(Invalid):p.summarize(plan,tmp/'out')

    def test_category_conditioned_and_main_brand_are_distinct(self):
        req=dict(operation='observation_factorial',left='sneakers',right='loafers',brand='Adidas',comparator='Reebok')
        cells=d.render(req,d.backgrounds('observation','pilot')[0])
        brand={k:int(v['brand_map']['S']=='Adidas') for k,v in cells.items()}
        conditional={k:int(k=='11') for k in cells}
        coefficients={'11':1,'10':-1,'01':-1,'00':1}
        self.assertEqual(d.contrast(brand,coefficients),0)
        self.assertEqual(brand['11']-brand['10'],1)
        self.assertEqual(d.contrast(conditional,coefficients),1)

    def test_failed_reservation_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,plan=method_plan(tmp)
            events=reg.events();reg._append('allocation',dict(j=plan['j']+1,contract_hash='failed',fingerprints=['failed']))
            panel=p.query_panel(dict(models=f.MODELS,generation=f.GEN),True)
            review=dict(plan_hash=panel['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='test')
            panel=c.allocate(reg,panel,review)
            self.assertEqual(panel['j'],plan['j']+2)
            receipt=tmp/'allocation.json';immutable_json(receipt,dict(j=panel['j'],hash=panel['hash']))
            env=dict(os.environ,SEEK_REPO=str(c.plans.ROOT),SEEK_DISCOVERY_MODE='panel',SEEK_DISCOVERY_ALLOCATION=str(receipt),SEEK_STUDY=str(reg.root),SEEK_DISCOVERY_OUTPUT=str(tmp/'out'))
            proc=subprocess.run(['bash',str(c.plans.ROOT/'seek_discovery.sh'),'--dry-run'],env=env,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertFalse((tmp/'out').exists())
            self.assertEqual(reg.events()[:len(events)],events)

    def test_missing_allocation_preflight(self):
        r=subprocess.run([sys.executable,str(c.plans.ROOT/'seek_discovery.py'),'preflight','--study','/missing','--allocation','/missing'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('blocked',r.stderr)

if __name__=='__main__':unittest.main()
