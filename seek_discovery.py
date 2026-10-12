#!/usr/bin/env python3
"""Small prospective discovery additions using the existing Seek runtime."""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json,immutable_json,row_lock
from seek.semantic_registry import Registry
from seek.discovery_phase import common as c,methods,panels,campaign,role_check,role_probe,role_followup,role_recovery,role_audit


def parser():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('incidents');a.add_argument('--row',required=True);a.add_argument('--study',required=True);a.add_argument('--rejections');a.add_argument('--allow-development-reuse',action='store_true');a.add_argument('--output',required=True)
    a=s.add_parser('role-check-prepare');a.add_argument('--study',required=True);a.add_argument('--allocation',required=True);a.add_argument('--index',type=int,required=True);a.add_argument('--qwen-config',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('role-check');a.add_argument('--plan',required=True);a.add_argument('--output',required=True);a.add_argument('--dry-run',action='store_true')
    a=s.add_parser('role-followup-prepare');a.add_argument('--probe-root',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('role-recovery-prepare');a.add_argument('--failed-root',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('role-audit');a.add_argument('--recovery-root',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('role-probe-prepare');a.add_argument('--study',required=True);a.add_argument('--allocation',required=True);a.add_argument('--role-plan',required=True);a.add_argument('--role-result',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('role-probe-review');a.add_argument('--plan',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('role-probe');a.add_argument('--plan',required=True);a.add_argument('--review',required=True);a.add_argument('--study',required=True);a.add_argument('--output',required=True);a.add_argument('--dry-run',action='store_true')
    a=s.add_parser('prepare');a.add_argument('--kind',choices=['method','observation','query-panel'],required=True);a.add_argument('--profile',required=True);a.add_argument('--incidents');a.add_argument('--output',required=True)
    a=s.add_parser('freeze');a.add_argument('--plan',required=True);a.add_argument('--study',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('preflight');a.add_argument('--study',required=True);a.add_argument('--allocation',required=True)
    a=s.add_parser('method');a.add_argument('--study',required=True);a.add_argument('--allocation',required=True);a.add_argument('--index',type=int,required=True);a.add_argument('--variant',choices=methods.VARIANTS,required=True);a.add_argument('--qwen-config',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('evaluate-prepare');a.add_argument('--study',required=True);a.add_argument('--allocation',required=True);a.add_argument('--runs',required=True);a.add_argument('--output',required=True)
    for cmd in ('panel','summarize'):
        a=s.add_parser(cmd);a.add_argument('--study',required=True);a.add_argument('--allocation',required=True);a.add_argument('--output',required=True)
        if cmd=='panel':a.add_argument('--model',choices=['query','observation','reference'],required=True)
    a=s.add_parser('pilot-gate');a.add_argument('--result',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('prompt-audit');a.add_argument('--files',nargs='+',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('simulate');a.add_argument('--output',required=True);a.add_argument('--replications',type=int,default=2);a.add_argument('--cap',type=int,default=64);a.add_argument('--seed',type=int,default=121012);a.add_argument('--schedule',choices=['fixed','adaptive'],default='adaptive');a.add_argument('--large',action='store_true')
    a=s.add_parser('supplement');a.add_argument('--recovery',required=True);a.add_argument('--original-archive',required=True);a.add_argument('--output',required=True)
    return p


def loaded(a):
    reg=Registry(a.study);receipt=read_json(a.allocation);j=receipt['j']
    if type(j) is not int or j<1:raise Invalid('invalid allocation receipt')
    plan=c.load(reg,j)
    if receipt['hash']!=plan['hash']:raise Invalid('allocation receipt mismatch')
    c.current(plan);return reg,plan


def gpu():
    if not os.environ.get('SLURM_JOB_ID'):raise Invalid('real inference requires user-submitted Slurm job')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def loader(entry,g):
    from seek.provenance import verify_weights
    from seek.victim import LegacyVictim
    verify_weights(entry);v=LegacyVictim(entry,g);v.simulated=False;return v


def main():
    a=parser().parse_args();result=None
    if a.cmd=='role-audit':
        result=role_audit.write(a.recovery_root,a.output)
    elif a.cmd=='role-recovery-prepare':
        draft=role_recovery.prepare(a.failed_root)
        print(json.dumps(dict(old_cap=draft['old_input_cap'],new_cap=draft['new_input_cap'],input_preflight=draft['input_preflight'],preserved_history_entries=len(draft['initial_history']),roles=draft['roles'],max_attempts=1,victim_calls=0,scope=draft['scope']),indent=2))
        if input('Approve the preserved final Goal request with a 10240-token input cap and one defender attempt? Type yes: ').strip().lower()!='yes':raise Invalid('recovery not approved')
        plan=role_recovery.approve(draft,getpass.getuser());immutable_json(a.output,plan)
        result=dict(status='prepared',plan=a.output,hash=plan['hash'],max_logical_calls=1,max_attempts=1,victim_calls=0,registry_mutations=0)
    elif a.cmd=='role-followup-prepare':
        plan=role_followup.prepare(a.probe_root);immutable_json(a.output,plan)
        result=dict(status='prepared',plan=a.output,hash=plan['hash'],probe_result_hash=plan['probe_result_hash'],roles=plan['roles'],max_logical_calls=4,max_attempts=8,victim_calls=0,registry_mutations=0,development_reuse=True)
    elif a.cmd=='role-probe-prepare':
        receipt=read_json(a.allocation);parent=c.load(Registry(a.study),receipt['j'])
        if parent['hash']!=receipt['hash']:raise Invalid('historical allocation receipt mismatch')
        plan=role_probe.prepare(parent,read_json(a.role_plan),read_json(a.role_result));immutable_json(a.output,plan)
        result=dict(status='prepared_requires_review',plan=a.output,hash=plan['hash'],request=plan['candidate']['request'],cells=plan['cells'],maximum_physical_responses=10,limitations=plan['limitations'])
    elif a.cmd=='role-probe-review':
        plan=read_json(a.plan);role_probe.validate(plan)
        print(json.dumps(dict(request=plan['candidate']['request'],authored_condition=plan['candidate']['condition'],profile=plan['profile'],cells=plan['cells'],anchor_ids=plan['anchor_ids'],limitations=plan['limitations'],maximum_physical_responses=10),indent=2))
        if input('Approve these development inputs, interpretation limits and maximum 10 victim responses? Type yes: ').strip().lower()!='yes':raise Invalid('not approved')
        result=dict(plan_hash=plan['hash'],accepted=True,reviewer=getpass.getuser(),reason='Reviewed exact authored contrast, inputs, development-only scope and ten-response budget')
        immutable_json(a.output,result)
    elif a.cmd=='role-probe':
        plan=read_json(a.plan);review=read_json(a.review);role_probe.check_review(plan,review)
        parent=c.load(Registry(a.study),plan['parent']['j'])
        if parent!=plan['parent']:raise Invalid('probe study parent mismatch')
        if a.dry_run:result=dict(status='ready',maximum_physical_responses=10,defender_calls=0,claim_allocations=0,scientific_confirmation=False)
        else:
            if plan['simulated']:raise Invalid('CLI probe requires real plan; fake backends belong in CPU tests')
            gpu()
            with row_lock(a.output):result=role_probe.run(plan,review,loader,Registry(a.study),a.output)
    elif a.cmd=='role-check-prepare':
        receipt=read_json(a.allocation);parent=c.load(Registry(a.study),receipt['j'])
        if parent['hash']!=receipt['hash']:raise Invalid('historical allocation receipt mismatch')
        config=read_json(a.qwen_config)
        agents=config.get('agents',config.get('defender',config))
        plan=role_check.prepare(parent,a.index,agents);immutable_json(a.output,plan)
        result=dict(status='prepared',plan=a.output,hash=plan['hash'],source_incident_index=a.index,
                    development_reuse=True,max_logical_calls=4,max_attempts=8,victim_calls=0,registry_mutations=0)
    elif a.cmd=='role-check':
        plan=read_json(a.plan)
        if plan.get('schema')=='seek-role-goal-recovery-v1':role_recovery.validate(plan)
        elif plan.get('schema')=='seek-role-evidence-followup-v1':role_followup.validate(plan)
        else:role_check.validate(plan)
        if a.dry_run:
            result=dict(status='ready',scope=plan['scope'],
                        development_reuse=True,max_logical_calls=plan['max_logical_calls'],max_attempts=plan['max_attempts'],victim_calls=0,registry_mutations=0)
        else:
            if plan['simulated']:raise Invalid('CLI role check requires real plan; fake backends belong in CPU tests')
            gpu()
            with row_lock(a.output):result=role_check.run(plan,methods.Roles(plan['agents']),a.output)
    elif a.cmd=='incidents':
        result=methods.select(a.row,Registry(a.study),a.allow_development_reuse,read_json(a.rejections) if a.rejections else None);immutable_json(a.output,result)
    elif a.cmd=='prepare':
        profile=read_json(a.profile)
        if a.kind=='method':
            if not a.incidents:raise Invalid('--incidents required')
            result=methods.compile_bank(read_json(a.incidents),profile)
        else:result=(panels.observation if a.kind=='observation' else panels.query_panel)(profile)
        immutable_json(a.output,result)
    elif a.cmd=='freeze':
        plan=c.checked(read_json(a.plan));print(json.dumps(plan,indent=2))
        if input('As independent reviewer, approve these inputs, scope, and frozen resource caps? Type yes: ').strip().lower()!='yes':raise Invalid('not approved')
        review=dict(plan_hash=plan['hash'],accepted=True,independent=True,resource_approved=True,reviewer=getpass.getuser(),reason='Reviewed full prospective inputs, evidence origin and frozen resource caps')
        result=c.allocate(Registry(a.study),plan,review);immutable_json(a.output,dict(j=result['j'],hash=result['hash']))
    elif a.cmd in ('preflight','method','evaluate-prepare','panel','summarize'):
        reg,plan=loaded(a)
        if a.cmd=='preflight':
            for entry in plan['profile']['models'].values():
                if digest(entry['weights'])!=entry['identity']:raise Invalid('model identity mismatch')
                if not plan['simulated'] and (not (Path(entry['path'])/'config.json').is_file() or any(not (Path(entry['path'])/n).is_file() for n in entry['weights'])):raise Invalid('missing checkpoint config/weight files')
            result=dict(status='ready',j=plan['j'],schema=plan['schema'],source_matches=True,model_calls=0)
        elif a.cmd=='method':
            if plan['schema']!='seek-query-method-v2':raise Invalid('method plan required')
            if a.index!=0 or a.variant!='adaptive_seek':
                gate=read_json(Path(a.output).parents[1]/'pilot_review.json')
                prefix=c.checked(read_json(Path(a.output).parents[1]/'incident-0/adaptive_seek/result.json'))
                if gate.get('plan_hash')!=plan['hash'] or gate.get('result_hash')!=prefix['hash'] or gate.get('accepted') is not True:raise Invalid('engineering pilot review required before remaining method matrix')
            gpu();config=read_json(a.qwen_config)
            # Same tested defender config format as seek_next.py bench-run.
            if 'agents' in config:config=config['agents']
            elif 'defender' in config:config=config['defender']
            with row_lock(a.output):result=methods.run(plan,a.index,a.variant,methods.Roles(config),loader,reg,a.output)
        elif a.cmd=='evaluate-prepare':
            runs=[read_json(Path(a.runs)/f'incident-{i}'/v/'result.json') for i in range(4) for v in methods.VARIANTS]
            result=methods.evaluation(plan,runs);immutable_json(a.output,result)
        elif a.cmd=='panel':
            if plan['schema'] not in ('seek-method-evaluator-v2','seek-observation-bridge-v1','seek-crossed-query-panel-v1'):raise Invalid('panel plan required')
            gpu()
            with row_lock(Path(a.output)/a.model):result=panels.run(plan,a.model,loader,reg,a.output)
        else:result=panels.summarize(plan,a.output)
    elif a.cmd=='pilot-gate':
        r=c.checked(read_json(a.result))
        if r['index']!=0 or r['variant']!='adaptive_seek' or r['status'] in ('backend_failure','victim_backend_failure','unscorable_probe') or not r['history']:raise Invalid('nonempty operational adaptive prefix required; abstention/unsupported hypotheses are acceptable, backend faults require separate development review')
        print(json.dumps(r,indent=2))
        if input('Review this engineering prefix and authorize the remaining frozen method matrix? Type yes: ').strip().lower()!='yes':raise Invalid('pilot not approved')
        result=dict(plan_hash=r['plan_hash'],result_hash=r['hash'],accepted=True,reviewer=getpass.getuser(),scope='engineering pilot review, not hypothesis approval or effect selection');immutable_json(a.output,result)
    elif a.cmd=='prompt-audit':result=panels.prompt_audit(a.files);immutable_json(a.output,result)
    elif a.cmd=='simulate':result=campaign.run(a.output,a.replications,a.cap,a.seed,a.schedule,a.large)
    elif a.cmd=='supplement':
        import tempfile,runpy
        from seek.victim import file_hash
        from seek.next_phase import audit
        recovery=Path(a.recovery);reports=[]
        for j in (7,8):
            folder=recovery/f'claim-{j:06d}';r=read_json(folder/'source_recovery.json');contract=read_json(folder/'frozen_contract.json')
            if r['status']!='frozen_source_map_matched' or r['frozen_hashes']!=contract['source']:raise Invalid('source supplement binding mismatch')
            if any(file_hash(folder/'source'/n)!=sha for n,sha in r['frozen_hashes'].items()):raise Invalid('recovered file changed')
            reports.append(dict(j=j,matching_commits=r['matching_commits'],files=len(r['frozen_hashes'])))
        original=file_hash(a.original_archive)
        scanner=runpy.run_path(str(ROOT/'docs/seek/package_verified_archive.py'))['credential_like']
        for path in recovery.rglob('*'):
            if path.is_symlink():raise Invalid('supplement symlink not allowed')
            if path.is_file() and (not audit.allowed(path.relative_to(recovery).as_posix()) or scanner(path.read_text(errors='replace'))):raise Invalid('unsafe supplement content')
        helper=runpy.run_path(str(ROOT/'docs/seek/package_audit.py'))
        result=helper['package'](recovery,a.output,paths=tuple(p.name for p in recovery.iterdir()),require_cluster=False)
        with tempfile.TemporaryDirectory() as tmp:audit.unpack(a.output,tmp)
        result.update(original_archive_sha256=original,sources=reports,scope='source byte matches only; historical partial audit unchanged',registry_mutations=0,model_calls=0)
        immutable_json(str(a.output)+'.addendum.json',result)
    print(json.dumps(result,indent=2))
    if a.cmd=='role-probe' and not a.dry_run and result['status']!='complete':return 2
    if isinstance(result,dict) and result.get('status') in ('prerequisite_missing','backend_failure','unscorable_cells','incomplete'):return 2
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (Invalid,KeyError,ValueError,OSError) as e:
        print(json.dumps(dict(status='blocked',reason=str(e),model_calls_not_inferred=True)),file=sys.stderr);sys.exit(2)
