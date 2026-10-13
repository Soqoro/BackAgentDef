"""Optional HACS commands in the existing seek_discovery entry point."""
import getpass
import json
from pathlib import Path
from ..schemas import Invalid
from ..storage import immutable_json, read_json, row_lock
from ..semantic_registry import Registry
from . import common as c, hacs as h, hacs_runtime as r, hacs_sim


def add_parser(sub):
    a=sub.add_parser('hacs-library');a.add_argument('--domain',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('hacs-prepare')
    for k in ('row','study','profile','qwen-config','library','output'):a.add_argument('--'+k,required=True)
    a.add_argument('--mode',choices=['pilot','matrix'],default='pilot');a.add_argument('--pilot-gate')
    a=sub.add_parser('hacs-freeze')
    for k in ('plan','study','output'):a.add_argument('--'+k,required=True)
    for cmd in ('hacs-run','hacs-preflight','hacs-evaluate','hacs-token-check'):
        a=sub.add_parser(cmd)
        for k in ('study','allocation'):a.add_argument('--'+k,required=True)
        if cmd!='hacs-preflight':a.add_argument('--output',required=True)
        if cmd=='hacs-run':
            a.add_argument('--index',type=int,required=True);a.add_argument('--variant',choices=h.VARIANTS,required=True);a.add_argument('--dry-run',action='store_true')
        if cmd=='hacs-evaluate':a.add_argument('--dry-run',action='store_true')
    a=sub.add_parser('hacs-gate');a.add_argument('--result',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('hacs-evaluation-prepare')
    for k in ('study','allocation','result','output'):a.add_argument('--'+k,required=True)
    a=sub.add_parser('hacs-evaluation-batch-prepare')
    for k in ('study','allocation','runs','output'):a.add_argument('--'+k,required=True)
    a=sub.add_parser('hacs-simulate');a.add_argument('--output',required=True);a.add_argument('--replications',type=int,default=2);a.add_argument('--cap',type=int,default=16);a.add_argument('--seed',type=int,default=131013);a.add_argument('--large',action='store_true')


def load(a):
    receipt=read_json(a.allocation);reg=Registry(a.study);plan=c.load(reg,receipt['j'])
    if plan['hash']!=receipt['hash']:raise Invalid('allocation_binding')
    c.current(plan)
    return reg,plan


def approval(prompt):
    if input(prompt+' Type yes: ').strip().lower()!='yes':raise Invalid('not_approved')


def _main(a,loader,gpu):
    if a.cmd=='hacs-library':
        domain=read_json(a.domain);family=h.hypotheses(domain,'pending_action_grounding');bank=h.bank(domain,family)
        result=c.seal(dict(schema='seek-hacs-library-freeze-v1',source=c.source(),domain=domain,capabilities=h.capability_card(domain),
                          family=family,bank=bank,coverage=h.coverage(bank,family),simulated=False,model_calls=0))
        immutable_json(a.output,result)
    elif a.cmd=='hacs-prepare':
        library=read_json(a.library);c.current(library)
        cfg=read_json(a.qwen_config);agents=cfg.get('agents',cfg.get('defender',cfg))
        result=r.prepare(a.row,Registry(a.study),read_json(a.profile),agents,library['domain'],a.mode,read_json(a.pilot_gate) if a.pilot_gate else None)
        result=c.seal(dict(result,library_hash=library['hash']));immutable_json(a.output,result)
    elif a.cmd=='hacs-freeze':
        plan=read_json(a.plan)
        if plan['schema']=='seek-hacs-plan-v1':r.validate_plan(plan)
        elif plan['schema']!='seek-hacs-evaluation-v1':raise Invalid('hacs_schema')
        print(json.dumps(dict(hash=plan['hash'],schema=plan['schema'],coverage=plan.get('coverage'),victim_budget=plan.get('victim_budget',len(plan.get('cells',[]))),
                              defender_budget=plan.get('defender_budget',0),anchors=plan['anchors'],scope=plan.get('scope','held-out finite support')),indent=2))
        approval('As independent reviewer, approve frozen library/inputs/caps and attest sources were collected after library freeze, outside prior exposure?')
        review=dict(plan_hash=plan['hash'],accepted=True,independent=True,resource_approved=True,reviewer=getpass.getuser(),reason='Reviewed prospective library, source freshness, exact inputs and caps')
        result=c.allocate(Registry(a.study),plan,review);immutable_json(a.output,dict(j=result['j'],hash=result['hash']))
        result=dict(status='allocated',j=result['j'],hash=result['hash'])
    elif a.cmd in ('hacs-run','hacs-preflight','hacs-evaluate','hacs-token-check'):
        reg,plan=load(a)
        if plan['schema']=='seek-hacs-plan-v1':r.validate_plan(plan)
        if a.cmd=='hacs-token-check':
            r.validate_plan(plan);receipts=[]
            for incident in plan['incidents']:
                family=h.hypotheses(plan['domain'],'pending_action_grounding')
                search=h.Search(plan['domain'],family,h.bank(plan['domain'],family))
                for role,state in [('Action',None),('Goal',search),('State',search)]:
                    counts=r.role_tokens.measure(plan['agents'],r.messages(role,r.packet(plan,incident,state)))
                    receipts.append(dict(case=incident['public']['case_id'],role=role,**counts,within_cap=counts['input_tokens']<=plan['input_cap']))
            result=dict(schema='seek-hacs-token-preflight-v1',plan_hash=plan['hash'],receipts=receipts,model_calls=0,scope='initial packets only; every actual later packet is separately preflighted')
            immutable_json(a.output,result)
            if not all(x['within_cap'] for x in receipts):raise Invalid('context_capacity_insufficient')
        elif a.cmd=='hacs-preflight':result=dict(status='ready',j=plan['j'],schema=plan['schema'],model_calls=0)
        elif a.cmd=='hacs-run':
            if plan['schema']!='seek-hacs-plan-v1' or a.variant not in plan['variants'] or not 0<=a.index<len(plan['incidents']):raise Invalid('invalid_run_index_or_variant')
            if a.dry_run:result=dict(status='ready',victim_budget=plan['victim_budget'],anchors=2,defender_budget=plan['defender_budget'],model_calls=0)
            else:
                if plan['simulated']:raise Invalid('real_cli_requires_real_population')
                gpu()
                with row_lock(a.output):result=r.run(plan,a.index,a.variant,r.Roles(plan['agents']),loader,reg,a.output)
        else:
            if plan['schema']!='seek-hacs-evaluation-v1':raise Invalid('evaluation_schema')
            if a.dry_run:result=dict(status='ready',logical_responses=len(plan['cells']),anchors=plan['anchors'],model_calls=0)
            else:
                gpu()
                with row_lock(a.output):result=(r.evaluate_batch if plan.get('members') else r.evaluate)(plan,loader,reg,a.output)
    elif a.cmd=='hacs-gate':
        result=c.checked(read_json(a.result));folder=Path(a.result).parent;plan=read_json(folder/'plan.json');r.validate_plan(plan)
        if result['plan_hash']!=plan['hash'] or plan['mode']!='pilot' or result['simulated'] or result['failure'] or result['status']=='backend_failure':raise Invalid('real_completed_pilot_required')
        print(json.dumps(read_json(folder/'review_packet.json'),indent=2))
        approval('Have you independently inspected the decision/evidence/review journals and found the new pilot engineering-valid?')
        assessment={k:input(k+' (yes/no): ').strip().lower()=='yes' for k in r.RUBRIC}
        approval('Approve four fresh groups x three variants, 384 discovery logical responses plus 24 anchors, with separate held-out review?')
        gate=c.seal(dict(pilot_result_hash=result['hash'],accepted=True,independent=True,resource_approved=True,
            reviewer=getpass.getuser(),reason='Independent prospective HACS pilot review',rubric=r.RUBRIC,assessment=assessment,method_matrix_unlocked=all(assessment.values())))
        r.check_gate(gate);immutable_json(a.output,gate);result=gate
    elif a.cmd=='hacs-evaluation-batch-prepare':
        _,plan=load(a);values=[]
        for i in range(len(plan['incidents'])):
            for variant in plan['variants']:
                values.append(c.checked(read_json(Path(a.runs)/('incident-'+str(i))/variant/'result.json')))
        count=sum(x.get('candidate') is not None and x.get('anchors_stable') is True and x.get('failure') is None for x in values)
        print(json.dumps(dict(denominator=len(values),primaries=count,logical_responses=count*24,anchors=count*2,
                             candidates=[dict(index=x['index'],variant=x['variant'],candidate=x['candidate']) for x in values]),indent=2))
        approval('As independent reviewer approve these unchanged primaries, common private finite support and stated evaluation caps?')
        review=dict(result_hashes=sorted(x['hash'] for x in values),accepted=True,independent=True,resource_approved=True,reviewer=getpass.getuser())
        result=r.evaluation_batch_plan(plan,values,review);immutable_json(a.output,result)
    elif a.cmd=='hacs-evaluation-prepare':
        _,plan=load(a);value=c.checked(read_json(a.result))
        print(json.dumps(dict(candidate=value.get('candidate'),responses=24,anchors=2,scope='complete fresh finite support; no certificate'),indent=2))
        approval('As independent reviewer, approve this unchanged primary and 24 held-out responses plus two anchors?')
        review=dict(result_hash=value['hash'],accepted=True,independent=True,resource_approved=True,reviewer=getpass.getuser())
        result=r.evaluation_plan(plan,value,review);immutable_json(a.output,result)
    else:result=hacs_sim.run(a.output,a.replications,a.cap,a.seed,a.large)
    print(json.dumps(result,indent=2))
    return 2 if result.get('failure') or result.get('status') in ('backend_failure','unscorable_probe','determinism_unresolved') else 0


def main(a,loader,gpu):
    try:return _main(a,loader,gpu)
    except (Invalid,KeyError,ValueError,TypeError,OSError) as exc:
        value=dict(schema='seek-hacs-command-failure-v1',status='blocked',error_type=type(exc).__name__,
                   reason=str(exc),scientific_confirmation=False,model_calls_not_inferred=True)
        output=getattr(a,'output',None)
        if output:
            path=Path(output)
            path=path.with_suffix(path.suffix+'.failure.json') if path.suffix else path/'failure.json'
            try:immutable_json(path,value)
            except (Invalid,OSError):pass  # Preserve any earlier failure artifact.
        print(json.dumps(value,indent=2));return 2
