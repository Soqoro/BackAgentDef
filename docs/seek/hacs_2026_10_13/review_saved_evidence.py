"""Bounded development review of frozen evidence: zero victim calls."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from audit_continuation import reconstruct, ROOT
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.discovery_phase import common as c, hacs as h, hacs_runtime as r
from seek.discovery_phase.hacs_review_evidence import display
from seek.storage import read_json, immutable_json, Journal, row_lock
from seek.schemas import Invalid


def prepare(run):
    root=Path(run); audit=reconstruct(root)
    if audit['status']!='local_evidence_reconstructed': raise Invalid('intact saved evidence required')
    old=c.checked(read_json(root/'plan.json')); result=c.checked(read_json(root/'result.json'))
    if result['status']!='development_complete': raise Invalid('completed development required')
    family=h.hypotheses(old['domain'],old['outcome']['id'])
    events=[c.checked(read_json(root/'evidence'/f'{i:03d}.json')) for i in range(result['completed_pairs'])]
    if any(set(e['predictions'])!={x['id'] for x in family['hypotheses']} for e in events):
        raise Invalid('historical family cannot be reconstructed')
    packets=[]
    for i,review in enumerate(result['reviews'],1):
        packet=copy.deepcopy(read_json(root/'packets'/f'{i:02d}.json'))
        count=next(n for n,e in enumerate(events,1) if e['id']==packet['factual_acknowledgment'][-1]['id'])
        selected,summary=display(family,events[:count],events[count-1]['after'])
        packet['rules']=[{k:x[k] for k in ('id','ast','origin','epoch')} for x in selected]
        packet['evidence_display']=summary
        packet['update']={field:{x['id']:events[count-1][field][x['id']] for x in selected} for field in ('predictions','losses')}
        packet['update']['weights']={x['id']:events[count-1]['after'][x['id']] for x in selected}
        packets.append(dict(role=review['role'],packet=packet))
    return c.seal(dict(schema='seek-saved-evidence-review-v1',source=c.source(),
        script_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_hash=audit['hash'],parent_result_hash=result['hash'],agents=old['agents'],
        outcome=old['outcome'],packets=packets,input_cap=4096,simulated=old['simulated'],
        development_reuse=True,victim_calls=0,registry_mutations=0,method_matrix_unlocked=False,
        defender_budget=len(packets),maximum_defender_attempts=2*len(packets)))


def validate(plan):
    c.current(plan)
    if plan['script_sha256']!=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(): raise Invalid('review script changed')
    if plan['schema']!='seek-saved-evidence-review-v1' or plan['input_cap']!=4096 or len(plan['packets'])!=4:
        raise Invalid('review contract mismatch')
    if plan['victim_calls']!=0 or plan['maximum_defender_attempts']!=8 or plan['defender_budget']!=4: raise Invalid('budget mismatch')


def token_check(plan):
    receipts=[]
    for i,item in enumerate(plan['packets'],1):
        # Measure both supported repair forms as well as each actual saved packet.
        payloads=[('initial',item['packet'])]+[(code,r.review_repair_packet(item['packet'],code,plan['outcome']['target']))
            for code in ('observed_json_array_required','observed_facts_mismatch')]
        for label,p in payloads:
            receipt=r.role_tokens.measure(plan['agents'],r.messages(item['role'],p))
            receipts.append(dict(review=i,packet=label,**receipt,within_cap=receipt['input_tokens']<=plan['input_cap']))
    return dict(plan_hash=plan['hash'],receipts=receipts,model_calls=0,ready=all(x['within_cap'] for x in receipts))


def run(plan,out):
    validate(plan)
    if not os.environ.get('SLURM_JOB_ID') or plan['simulated']: raise Invalid('real Slurm allocation required')
    out=Path(out)
    with row_lock(out):
        if (out/'started.json').exists(): raise Invalid('existing review output; preserve it')
        check=token_check(plan)
        if not check['ready']: raise Invalid('packet exceeds cap; do not raise cap')
        immutable_json(out/'plan.json',plan);immutable_json(out/'preflight.json',check)
        immutable_json(out/'started.json',dict(plan_hash=plan['hash']))
        backend=r.Roles(plan['agents']); journal=Journal(out/'defender')
        reviews=[];failure=None
        try:
            for i,item in enumerate(plan['packets'],1):
                value=r.review_call(plan,backend,item['role'],item['packet'],out,journal,i,plan['outcome']['target'])
                reviews.append(value);immutable_json(out/'reviews'/f'{i:02d}.json',value)
        except Exception as exc: failure=dict(type=type(exc).__name__,reason=str(exc))
        finally: backend.close()
        result=c.seal(dict(schema='seek-saved-evidence-review-result-v1',plan_hash=plan['hash'],
            parent_result_hash=plan['parent_result_hash'],status='backend_failure' if failure else 'reviews_complete',
            reviews=reviews,failure=failure,defender_costs=r.token_cost(out,'defender'),
            simulated=False,development_reuse=True,victim_calls=0,registry_mutations=0,
            independent=False,method_matrix_unlocked=False,scientific_confirmation=False,
            scope='post-inspection reviews of saved evidence; no new probes or automatic qualitative pass'))
        immutable_json(out/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','token-check','run'])
    parser.add_argument('--run-root');parser.add_argument('--plan',required=True)
    parser.add_argument('--output');parser.add_argument('--approve-development',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        if not args.run_root: parser.error('--run-root required')
        plan=prepare(args.run_root);immutable_json(args.plan,plan)
        print(json.dumps(dict(status='prepared',plan_hash=plan['hash'],victim_calls=0,maximum_defender_attempts=8)))
    else:
        plan=c.checked(read_json(args.plan));validate(plan)
        if not args.output: parser.error('--output required')
        if args.command=='token-check':
            result=token_check(plan);immutable_json(args.output,result)
        else:
            if not args.approve_development: parser.error('--approve-development required')
            result=run(plan,args.output)
        print(json.dumps(result,indent=2))
        if result.get('ready') is False or result.get('failure'): sys.exit(2)
