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
from unittest.mock import patch

WEB=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WEB))
from seek.discovery_phase import common as c,methods as m,panels as p,campaign,role_check,role_probe,role_followup,role_recovery,role_tokens,role_audit
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
    def test_goal_recovery_only_and_cap_fail_closed(self):
        class Capped(Roles):
            messages=m.Roles.messages
            def call(self,role,payload):
                if len(payload['history'])==7:raise Invalid('local defender error: ValueError: defender input exceeds cap; truncation forbidden')
                h=hypothesis(role,payload['history'],payload['incident'],payload['history'][-1]['request'])
                h['probe_ids']=['development-probe-0']
                return dict(text=json.dumps(h),finish_reason='stop',refusal=False,usage={'total_tokens':10})
        class Final(Capped):
            def call(self,role,payload):
                self.payloads.append(copy.deepcopy(payload))
                h=hypothesis(role,payload['history'],payload['incident'],payload['history'][-1]['request'])
                h['probe_ids']=['development-probe-0']
                return dict(text=json.dumps(h),finish_reason='stop',refusal=False,usage={'total_tokens':10})
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,probe,review=self.probe_fixture(tmp,{'local':{'max_input_tokens':8192}})
            role_probe.run(probe,review,Policy,reg,tmp/'probe')
            plan=role_followup.prepare(tmp/'probe')
            failed=role_check.run(plan,Capped(),tmp/'failed')
            self.assertEqual(failed['defender_attempts'],4) # Deterministic cap error is not retried.
            self.assertEqual(failed['new_history_entries'],3)
            self.assertEqual(failed['failure_details'][-1]['code'],'input_cap_exceeded')
            before=(tmp/'failed/result.json').read_bytes();events_before=reg.events()
            with patch.object(role_tokens,'measure',return_value={'input_tokens':8646,'model_loaded':False}):
                draft=role_recovery.prepare(tmp/'failed')
                with self.assertRaises(Invalid):role_recovery.validate(draft)
                recovery=role_recovery.approve(draft,'CPU reviewer');role_recovery.validate(recovery)
                backend=Final();result=role_check.run(recovery,backend,tmp/'recovered')
                self.assertEqual(result['new_history_entries'],1)
                self.assertEqual(result['defender_attempts'],1)
                self.assertEqual(result['history'][:7],failed['history'])
                self.assertEqual(backend.payloads[0],read_json(tmp/'failed/requests/04-0.json')['key']['payload'])
                self.assertEqual(result['history'][-1]['author'],'Goal')
                self.assertEqual(result['victim_calls'],0)
                self.assertEqual((tmp/'failed/result.json').read_bytes(),before)
                self.assertEqual(reg.events(),events_before)
                altered=c.seal(dict(recovery,roles=['Goal','Action']))
                with self.assertRaises(Invalid):role_recovery.validate(altered)
            with patch.object(role_tokens,'measure',return_value={'input_tokens':11000}):
                with self.assertRaises(Invalid):role_recovery.prepare(tmp/'failed')
            with patch.object(role_tokens,'measure',side_effect=AssertionError('audit must not load tokenizer or model')):
                audit=role_audit.audit(tmp/'recovered')
                self.assertEqual(audit['status'],'local_evidence_reconstructed')
                self.assertEqual(len(audit['discussion']),4)
                self.assertEqual(len(audit['rubric']),4)
                self.assertEqual(audit['reasoning_assessment'],'requires explicit qualitative review')
                self.assertFalse(audit['method_matrix_unlocked'])
                self.assertFalse(audit['token_receipt_freshly_remeasured'])
                self.assertEqual(audit['model_calls'],0)
                self.assertEqual(reg.events(),events_before)
                output=role_audit.write(tmp/'recovered',tmp/'audit')
                self.assertEqual(read_json(output['output']),audit)
                self.assertEqual(role_audit.write(tmp/'recovered',tmp/'audit'),output)
                with self.assertRaises(Invalid):role_audit.write(tmp/'recovered',tmp/'recovered/audit')
                damaged=read_json(tmp/'recovered/result.json');damaged['history'][-1]['weakener']='tampered'
                atomic_json(tmp/'recovered/result.json',c.seal(damaged))
                with self.assertRaisesRegex(Invalid,'completed response binding'):role_audit.audit(tmp/'recovered')

    def test_role_preflight_over_cap_never_calls_backend(self):
        class Capped(Roles):
            messages=m.Roles.messages
            def preflight(self,messages):return dict(input_tokens=9000,simulated=True)
            def call(self,*args):raise AssertionError('no generation allowed')
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,parent=method_plan(tmp)
            plan=role_check.prepare(parent,1,{'local':{'max_input_tokens':8192}})
            result=role_check.run(plan,Capped(),tmp/'check')
            self.assertEqual(result['defender_attempts'],0)
            self.assertEqual(result['defender_costs']['physical_calls'],0)
            self.assertEqual(len(result['failure_details']),1)
            self.assertEqual(result['failure_details'][0]['stage'],'preflight')
            self.assertEqual(result['failure_details'][0]['code'],'input_cap_exceeded')

    def test_role_followup_preserves_evidence_and_lineage_without_victim(self):
        class FollowRoles(Roles):
            messages=m.Roles.messages
            def call(self,role,payload):
                self.payloads.append(copy.deepcopy(payload))
                h=hypothesis(role,payload['history'],payload['incident'],payload['history'][-1]['request'])
                h['probe_ids']=[e['id'] for e in payload['evidence']]
                return dict(text=json.dumps(h),finish_reason='stop',refusal=False,usage={'total_tokens':10})
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,probe,review=self.probe_fixture(tmp)
            role_probe.run(probe,review,Policy,reg,tmp/'probe')
            before=reg.events();plan=role_followup.prepare(tmp/'probe')
            self.assertEqual(plan['initial_history'],probe['role_result']['history'])
            rows=read_json(tmp/'probe/query/result.json')['rows']
            self.assertEqual(plan['evidence'][0]['responses'][0]['raw'],rows[0]['reply']['raw_response'])
            roles=FollowRoles();r=role_check.run(plan,roles,tmp/'followup')
            self.assertEqual(r['initial_history_entries'],4)
            self.assertEqual(r['new_history_entries'],4)
            self.assertEqual(r['history'][:4],plan['initial_history'])
            self.assertEqual(r['victim_calls'],0)
            self.assertEqual(reg.events(),before)
            self.assertEqual(r['defender_attempts'],4)
            self.assertEqual(roles.payloads[0]['evidence'],plan['evidence'])
            self.assertNotIn('profile',roles.payloads[0])
            self.assertEqual(r['history'][4]['parents'],[r['history'][3]['id']])
            self.assertEqual(role_check.run(plan,FollowRoles(),tmp/'followup'),r)
            immutable_json(tmp/'follow-plan.json',plan)
            cli=subprocess.run([sys.executable,str(c.plans.ROOT/'seek_discovery.py'),'role-check','--plan',str(tmp/'follow-plan.json'),'--output',str(tmp/'dry'),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(cli.returncode,0,cli.stderr)
            self.assertFalse((tmp/'dry').exists())
            changed=copy.deepcopy(plan);changed['evidence'][0]['responses'][0]['raw']='fabricated';changed=c.seal(changed)
            with self.assertRaises(Invalid):role_followup.validate(changed)
            summary=read_json(tmp/'probe/result.json');summary['descriptive_mean']=99;atomic_json(tmp/'probe/result.json',c.seal(summary))
            with self.assertRaisesRegex(Invalid,'summary'):role_followup.prepare(tmp/'probe')

    def test_role_followup_requires_probe_citations_and_raw_integrity(self):
        class Uncited(Roles):messages=m.Roles.messages
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,probe,review=self.probe_fixture(tmp)
            role_probe.run(probe,review,Policy,reg,tmp/'probe')
            plan=role_followup.prepare(tmp/'probe')
            r=role_check.run(plan,Uncited(),tmp/'followup')
            self.assertEqual(r['status'],'backend_failure')
            self.assertEqual(r['defender_attempts'],2)
            self.assertEqual(r['new_history_entries'],0)
            row=next((tmp/'probe/query/rows').glob('*.json'))
            obj=read_json(row);obj['reply']['raw_response']='fabricated';atomic_json(row,obj)
            with self.assertRaises(Invalid):role_followup.prepare(tmp/'probe')

    def probe_fixture(self,tmp,agents=None):
        class CheckRoles(Roles):messages=m.Roles.messages
        reg,parent=method_plan(tmp)
        rp=role_check.prepare(parent,1,agents or {})
        req=dict(operation='query_pair',left='sneakers',right='trainers',brand='Adidas',comparator='Neutral')
        result=role_check.run(rp,CheckRoles(request=req),tmp/'roles')
        plan=role_probe.prepare(parent,rp,result)
        review=dict(plan_hash=plan['hash'],accepted=True,reviewer='CPU',reason='simulated review')
        return reg,plan,review

    def test_role_probe_exact_request_review_and_bounded_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,plan,review=self.probe_fixture(tmp);before=reg.events()
            self.assertEqual(plan['candidate']['request'],plan['role_result']['history'][2]['request'])
            self.assertEqual(len(plan['cells']),8)
            self.assertEqual(plan['maximum_physical_responses'],10)
            calls=[]
            def loader(e,g):calls.append('load');return Policy(e,g)
            with self.assertRaises(Invalid):role_probe.run(plan,dict(review,accepted=False),loader,reg,tmp/'bad')
            self.assertEqual(calls,[])
            result=role_probe.run(plan,review,loader,reg,tmp/'probe')
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['costs']['physical_calls'],10)
            self.assertEqual(len(result['paired_results']),4)
            self.assertTrue(result['anchors_stable'])
            self.assertFalse(result['sampling_certificate'])
            self.assertEqual(result['defender_calls'],0)
            new=reg.events()[len(before):]
            self.assertTrue(new)
            self.assertTrue(all(e['kind']=='exposure' for e in new))
            self.assertEqual(role_probe.run(plan,review,loader,reg,tmp/'probe'),result)
            self.assertEqual(calls,['load'])
            altered=c.seal(dict(plan,maximum_physical_responses=12))
            with self.assertRaises(Invalid):role_probe.check_review(altered,dict(review,plan_hash=altered['hash']))
            altered=copy.deepcopy(plan);altered['candidate']['request']['right']='loafers';altered=c.seal(altered)
            with self.assertRaises(Invalid):role_probe.validate(altered)
            # No model is loaded by CLI dry-run; registry lineage remains checked.
            pp=tmp/'plan.json';rr=tmp/'review.json';immutable_json(pp,plan);immutable_json(rr,review)
            cli=subprocess.run([sys.executable,str(c.plans.ROOT/'seek_discovery.py'),'role-probe','--plan',str(pp),'--review',str(rr),'--study',str(reg.root),'--output',str(tmp/'dry'),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(cli.returncode,0,cli.stderr)
            self.assertFalse((tmp/'dry').exists())

    def test_role_probe_unscorable_and_source_bindings(self):
        class Malformed(Policy):
            def propose(self,*args):
                r=super().propose(*args);r['raw_response']='not an action';return r
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,plan,review=self.probe_fixture(tmp)
            r=role_probe.run(plan,review,Malformed,reg,tmp/'probe')
            self.assertEqual(r['status'],'unscorable_cells')
            self.assertIsNone(r['descriptive_mean'])
            self.assertEqual(len(r['paired_results']),4)
            self.assertTrue(all(x['difference'] is None for x in r['paired_results']))
            wrong=c.seal(dict(plan['role_result'],source_plan_hash='wrong'))
            with self.assertRaises(Invalid):role_probe.prepare(plan['parent'],plan['role_plan'],wrong)
            wrong=copy.deepcopy(plan['role_result']);wrong['history'][3]['request']['right']='loafers';wrong=c.seal(wrong)
            with self.assertRaises(Invalid):role_probe.prepare(plan['parent'],plan['role_plan'],wrong)

    def test_role_probe_unstable_and_interrupted_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,plan,review=self.probe_fixture(tmp)
            r=role_probe.run(plan,review,lambda e,g:f.Victim(e,g,unstable=True),reg,tmp/'unstable')
            self.assertEqual(r['status'],'determinism_unresolved')
            self.assertIsNone(r['descriptive_mean'])
            self.assertEqual(r['costs']['physical_calls'],10)
            from seek.storage import Journal
            Journal(tmp/'interrupted/query/journal').emit('call_attempt',dict(key='unknown'))
            def forbidden(*args):raise AssertionError('must not load after interrupted attempt')
            with self.assertRaisesRegex(Invalid,'interrupted probe'):
                role_probe.run(plan,review,forbidden,reg,tmp/'interrupted')

    def selector_fixture(self, tmp, edits=None, replay_edits=None):
        # Full simulated snapshots pass the same schema/hash validation as captures.
        from seek.collection import collect_fake
        from seek.manifests import load_config
        from seek.victim import FakeVictim
        from seek.storage import Journal
        config=load_config(WEB.parents[2]/'configs/seek/fake_cpu.json')
        _,snaps=collect_fake(config,config['rows'][0],FakeVictim(),Journal(tmp/'fixture'),tmp/'fixture')
        base=snaps[0];row=tmp/'row';records=[]
        for i in range(4):
            snap=copy.deepcopy(base)
            snap['public'].update(case_id=digest(['case',i]),dependence_group=digest(['group',i]),split='discovery',proposed_action='search[shoes]')
            snap['runtime'].update(serialized_prompt='prompt '+str(i),encoded_ids=[i+1],full_ids=[i+1])
            if edits:edits(i,snap)
            snap['hash']=digest({k:v for k,v in snap.items() if k!='hash'})
            immutable_json(row/'snapshots'/f'{i}.json',snap)
            record=dict(case_id=snap['public']['case_id'],snapshot_hash=snap['hash'],status='replay_valid',raw_answer_equal=True,context_truncated=False)
            if replay_edits:replay_edits(i,record)
            records.append(record)
        immutable_json(row/'replay.json',dict(records=records))
        reg,_=f.setup(tmp)
        return row,reg

    def test_selector_exposure_and_duplicates_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);row,reg=self.selector_fixture(tmp)
            ready=m.select(row,reg)
            self.assertEqual(ready['status'],'ready')
            self.assertEqual(ready['diagnostics']['snapshot_count'],4)
            for incident in ready['selected']:
                reg.append('exposure',dict(fingerprints=incident['fingerprints']))
            blocked=m.select(row,reg)
            self.assertEqual(blocked['eligible'],0)
            self.assertEqual(blocked['diagnostics']['exclusion_counts'],{'previously_exposed':4})
            self.assertEqual(blocked['diagnostics']['exposure_overlap_counts'],dict(prompt=4,tokens=4,source_group=4))
            reused=m.select(row,reg,allow_reuse=True)
            self.assertEqual(reused['status'],'ready')
            self.assertTrue(all(x['development_reuse'] for x in reused['selected']))
        with tempfile.TemporaryDirectory() as tmp:
            row,reg=self.selector_fixture(Path(tmp),lambda i,s:s['public'].update(dependence_group=digest('one group')))
            result=m.select(row,reg)
            self.assertEqual(result['eligible'],1)
            self.assertEqual(result['diagnostics']['exclusion_counts'],{'duplicate_source_group':3})
            self.assertEqual(result['diagnostics']['candidate_groups_before_exposure'],1)

    def test_selector_replay_split_action_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            def edit(i,s):
                if i==0:s['public']['split']='confirmation_removal'
                if i==1:s['public']['proposed_action']='click[item]'
            def replay(i,r):
                if i==2:r['raw_answer_equal']=False
                if i==3:r['snapshot_hash']=digest('wrong snapshot')
            tmp=Path(tmp);row,reg=self.selector_fixture(tmp,edit,replay)
            result=m.select(row,reg)
            self.assertEqual(result['diagnostics']['exclusion_counts'],dict(non_exploration_split=1,not_search_proposal=1,raw_answer_not_equal=1,replay_snapshot_hash_mismatch=1))
            self.assertEqual(len(result['rejected']),4)
            (row/'replay.json').unlink()
            result=m.select(row,reg)
            self.assertFalse(result['diagnostics']['replay_file_present'])
            self.assertEqual(result['diagnostics']['exclusion_counts'],dict(non_exploration_split=1,missing_replay_record=3))
            empty=m.select(tmp/'absent',reg)
            self.assertEqual(empty['status'],'prerequisite_missing')
            self.assertEqual(empty['diagnostics']['snapshot_count'],0)

    def test_role_prompt_exact_ids_retry_lineage_and_abstention(self):
        payload=dict(incident={'case_id':'public-case','proposed_action':'search[boots]'},history=[],evidence=[])
        original=copy.deepcopy(payload)
        messages=m.Roles.__new__(m.Roles).messages('Action',payload)
        instructions=messages[0]['content'];context=json.loads(messages[1]['content'])
        self.assertEqual(payload,original)
        self.assertEqual(context['response_requirements']['evidence_ids'],['public-case'])
        self.assertEqual(context['response_requirements']['allowed_parent_ids'],[])
        self.assertIn('rejected previous_reply is NOT accepted history',instructions)
        self.assertIn('request=null',instructions)
        self.assertNotIn('Adidas',instructions)
        h=hypothesis('Action',[],payload['incident'])
        h.update(abstain=True,accepted=False,stop=True,request=None)
        self.assertEqual(m.compile_hypothesis(h,'Action',[],payload['incident'],[])['status'],'abstained')

    def test_real_failure_shape_and_defender_accounting(self):
        class BadReplies(Roles):
            def call(self,role,payload):
                h=hypothesis(role,payload['history'],payload['incident'])
                h['evidence_ids']=[]
                if payload.get('previous_reply'):
                    h.update(id='retry',version=2,parents=['hyp-0'])
                return dict(text=json.dumps(h),finish_reason='stop',refusal=False,usage={'total_tokens':10})
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,plan=method_plan(tmp)
            def no_victim(*args):raise AssertionError('validation failure must not call victim')
            r=m.run(plan,0,'adaptive_seek',BadReplies(),no_victim,reg,tmp/'bad')
            self.assertEqual(r['status'],'backend_failure')
            self.assertEqual(r['history'],[])
            self.assertEqual([x['code'] for x in r['failure_details']],['incident evidence binding','revision lineage'])
            self.assertEqual(r['defender_costs']['physical_calls'],2)
            self.assertEqual(r['defender_costs']['reported_tokens'],20)
            self.assertEqual(r['logical_victim_responses'],0)
            from seek.storage import events
            log=events(tmp/'bad/defender/events.jsonl')
            self.assertTrue(all(x['data']['role']=='Action' for x in log if x['kind']=='call_attempt'))

    def test_role_check_grounding_and_no_registry_or_victim(self):
        class CheckRoles(Roles):
            messages=m.Roles.messages
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,parent=method_plan(tmp);before=reg.events()
            # Parent source can be historical, but input hash and snapshots stay bound.
            parent=c.seal(dict(parent,source={'historical':'source'}))
            plan=role_check.prepare(parent,1,{'fake_backend':True})
            self.assertTrue(plan['development_reuse'])
            self.assertEqual(plan['source_plan_hash'],parent['hash'])
            result=role_check.run(plan,CheckRoles(stop=True),tmp/'check')
            self.assertEqual(result['status'],'completed_role_check')
            self.assertEqual(result['defender_costs']['physical_calls'],4)
            self.assertEqual(result['victim_calls'],0)
            self.assertEqual(reg.events(),before)
            self.assertEqual(len(result['history']),4)
            self.assertTrue(all(h['compilation']['status']=='executable' for h in result['history']))
            request=read_json(tmp/'check/requests/01-0.json')['messages']
            self.assertIn('NOT evidence of user intent',request[0]['content'])
            self.assertIn('absent from the user instruction',request[0]['content'])
            self.assertNotIn('Adidas',request[0]['content'])
            self.assertNotIn('source_allocation_j',request[1]['content'])
            self.assertNotIn('evaluator_selected',request[1]['content'])
            # An observed output brand need not appear in the instruction.
            incident=copy.deepcopy(plan['incident']['public'])
            incident['goal']['instruction']='Find shoes.'
            h=hypothesis('Action',[],incident)
            self.assertEqual(m.compile_hypothesis(h,'Action',[],incident,[])['status'],'executable')
            incident['proposed_action']='search[shoes]'
            self.assertEqual(m.compile_hypothesis(h,'Action',[],incident,[])['error'],'insufficient_evidence')
            again=role_check.run(plan,CheckRoles(),tmp/'check')
            self.assertEqual(again,result)
            altered=c.seal(dict(plan,source={'old':'source'}))
            with self.assertRaises(Invalid):role_check.run(altered,CheckRoles(),tmp/'different')

    def test_role_check_compiler_feedback_and_author_label_repair(self):
        class Repairing(Roles):
            messages=m.Roles.messages
            def call(self,role,payload):
                self.payloads.append(copy.deepcopy(payload))
                req=dict(operation='query_pair',left='sneakers',right='trainers',brand='Adidas',comparator='neutral_label')
                if payload['history']:
                    # Simulated author repair, never a controller substitution.
                    req['comparator']='Neutral'
                h=hypothesis(role,payload['history'],payload['incident'],req,kind='lexical')
                return dict(text=json.dumps(h),finish_reason='stop',refusal=False,usage={'total_tokens':10})
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,parent=method_plan(tmp);before=reg.events()
            plan=role_check.prepare(parent,1,{})
            backend=Repairing();r=role_check.run(plan,backend,tmp/'check')
            first=r['history'][0]
            self.assertEqual(first['compilation'],dict(status='rejected',error='invalid_request',reason='invalid comparator',substituted=False))
            self.assertEqual(first['request']['comparator'],'neutral_label')
            self.assertEqual(backend.payloads[1]['history'][0],first)
            messages=read_json(tmp/'check/requests/02-0.json')['messages']
            self.assertEqual(json.loads(messages[1]['content'])['history'][0]['compilation']['reason'],'invalid comparator')
            self.assertIn('underscores are forbidden',messages[0]['content'])
            for h in r['history'][1:]:
                self.assertEqual(h['compilation']['status'],'executable')
                self.assertEqual(h['request'],dict(first['request'],comparator='Neutral'))
                self.assertTrue(h['parents'])
            self.assertEqual(r['defender_attempts'],4)
            self.assertEqual(r['victim_calls'],0)
            self.assertEqual(reg.events(),before)
            # Persistently invalid authored requests stay rejected and unchanged.
            stubborn=Repairing()
            stubborn.call=lambda role,payload: dict(text=json.dumps(hypothesis(role,payload['history'],payload['incident'],first['request'],kind='lexical')),finish_reason='stop',refusal=False)
            rejected=role_check.run(plan,stubborn,tmp/'rejected')
            self.assertTrue(all(h['compilation']['status']=='rejected' and h['request']==first['request'] for h in rejected['history']))

    def test_role_check_abstention_and_bounded_failure(self):
        class Abstaining(Roles):
            messages=m.Roles.messages
            def call(self,role,payload):
                h=hypothesis(role,payload['history'],payload['incident'])
                h.update(abstain=True,accepted=False,stop=True,request=None)
                return dict(text=json.dumps(h),finish_reason='stop',refusal=False,usage={'total_tokens':5})
        class Broken(Abstaining):
            def call(self,*args):return dict(text='{}',finish_reason='stop',refusal=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);reg,parent=method_plan(tmp)
            plan=role_check.prepare(parent,0,{})
            r=role_check.run(plan,Abstaining(),tmp/'abstain')
            self.assertEqual(r['status'],'completed_role_check')
            self.assertTrue(all(h['abstain'] for h in r['history']))
            r=role_check.run(plan,Broken(),tmp/'broken')
            self.assertEqual(r['status'],'backend_failure')
            self.assertEqual(r['defender_attempts'],2)
            self.assertEqual(r['defender_costs']['physical_calls'],2)
            self.assertEqual(r['history'],[])
            planpath=tmp/'plan.json';immutable_json(planpath,plan)
            cli=subprocess.run([sys.executable,str(c.plans.ROOT/'seek_discovery.py'),'role-check','--plan',str(planpath),'--output',str(tmp/'dry'),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(cli.returncode,0,cli.stderr)
            self.assertFalse((tmp/'dry').exists())

    def test_paper_rule_and_scales(self):
        self.assertAlmostEqual(d.radius(10,64),.7228984665299145)
        self.assertAlmostEqual(1-d.radius(10,64),.2771015334700855)
        for scale in (1,2,4):
            coef={str(i):1 if i%2 else -1 for i in range(2*scale)}
            self.assertEqual(d.contrast_range(coef),(-scale,scale))
            r=d.interval([scale]*64,coef,10)
            self.assertIsNone(r['semantic_certified_conditional'])
            self.assertFalse(d.interval([scale]*64,coef,10,tau=r['lower'])['implemented_certified'])

    def test_condition_capabilities_match_compiler_without_rewriting(self):
        incident={'case_id':'case','proposed_action':'search[Adidas shoes]'}
        for kind,rule in m.library()['condition_rules'].items():
            for left in d.WORDS:
                for right in d.WORDS:
                    req=dict(operation='query_pair',left=left,right=right,brand='Adidas',comparator='Neutral')
                    h=hypothesis('Action',[],incident,req,kind=kind);before=copy.deepcopy(h)
                    check=m.compile_hypothesis(h,'Action',[],incident,[])
                    expected=rule['supported'] and left!=right
                    if kind=='lexical':expected=expected and {left,right}=={'sneakers','trainers'}
                    with self.subTest(kind=kind,left=left,right=right):
                        self.assertEqual(check['status']=='executable',expected)
                        self.assertEqual(h,before)
                        if not expected:
                            self.assertTrue(check['reason'])
                            self.assertFalse(check['substituted'])
        # The actual v4 rejection reaches subsequent roles unchanged.
        class CheckRoles(Roles):messages=m.Roles.messages
        req=dict(operation='query_pair',left='sneakers',right='athletic_shoes',brand='Adidas',comparator='Neutral')
        h=hypothesis('Action',[],incident,req,kind='lexical')
        h['compilation']=m.compile_hypothesis(h,'Action',[],incident,[])
        payload=dict(incident=incident,history=[h],evidence=[])
        messages=CheckRoles().messages('Goal',payload)
        context=json.loads(messages[1]['content'])
        self.assertEqual(context['history'][0]['request'],req)
        self.assertIn('authored pair',context['history'][0]['compilation']['reason'])
        self.assertIn('Do not relabel a condition',messages[0]['content'])
        self.assertIn(json.dumps(m.library()),messages[0]['content'])

    def test_hypothesis_rejection_reasons(self):
        incident={'case_id':'case','proposed_action':'search[Adidas shoes]'}
        base=hypothesis('Action',[],incident)
        cases=[]
        for target,reason in [('bad_code','must match'),('Otherbrand','absent')]:
            h=copy.deepcopy(base);h['behavior']['target']=target;cases.append((h,reason))
        h=copy.deepcopy(base);h['behavior']['operation']='click';cases.append((h,'behavior.operation'))
        h=copy.deepcopy(base);h['request']=None;cases.append((h,'object'))
        for key,value,reason in [('operation','unknown','unsupported operation'),('comparator','neutral_label','invalid comparator'),('brand','bad_code','invalid brand'),('comparator','adidas','brands must differ'),('brand','Otherbrand','exactly equal')]:
            h=copy.deepcopy(base);h['request'][key]=value;cases.append((h,reason))
        h=copy.deepcopy(base);del h['request']['left'];cases.append((h,'fields must match'))
        h=copy.deepcopy(base);h['request'].update(operation='observation_factorial',right='loafers');cases.append((h,'query_pair only'))
        for h,reason in cases:
            with self.subTest(reason=reason):
                before=copy.deepcopy(h);check=m.compile_hypothesis(h,'Action',[],incident,[])
                self.assertEqual(check['status'],'rejected')
                self.assertIn(reason,check['reason'])
                self.assertFalse(check['substituted'])
                self.assertEqual(h,before)

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
