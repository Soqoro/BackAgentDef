"""Reviewed single-Goal recovery; preserve failed evidence and accepted replies."""
import copy
import json
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import read_json, events
from . import common as c, methods, role_check, role_followup, role_tokens


def prepare(failed_root, counter=None):
    root=Path(failed_root);old=c.checked(read_json(root/'plan.json'));result=c.checked(read_json(root/'result.json'))
    derived=role_followup.prepare(old['probe_root'])
    if c.seal(dict(derived,source=old['source']))!=old:raise Invalid('historical followup plan mismatch')
    if (result['schema']!='seek-role-evidence-followup-result-v1' or result['plan_hash']!=old['hash'] or
        result['status']!='backend_failure' or result['new_history_entries']!=3 or result['initial_history_entries']!=4 or
        len(result['history'])!=7 or result['history'][:4]!=old['initial_history'] or result['victim_calls']!=0 or result['registry_mutations']!=0):
        raise Invalid('recovery requires exactly three accepted continuation replies and missing final Goal')
    records=events(root/'defender/events.jsonl')
    for i,e in enumerate(records):
        if e['sequence']!=i or e['id']!=digest([i,e['kind'],e['data']]):raise Invalid('mutated defender journal')
    for call,h in enumerate(result['history'][4:],1):
        authored={k:v for k,v in h.items() if k not in ('compilation','experiment_status')}
        check=methods.compile_hypothesis(authored,old['roles'][call-1],result['history'][:call+3],old['incident']['public'],[e['id'] for e in old['evidence']])
        if check!=h['compilation'] or not h['probe_ids']:raise Invalid('accepted reply compilation/citation mismatch')
        matched=False
        for request_path in sorted((root/'requests').glob(f'{call:02d}-*.json')):
            request=read_json(request_path)
            if 'key' not in request:continue
            key=request['key'];payload=key['payload']
            if key['plan']!=old['hash'] or payload['history']!=result['history'][:call+3]:raise Invalid('saved accepted history mismatch')
            for event in records:
                if event['kind']=='call_complete' and event['data']['key']==digest(key):
                    if json.loads(event['data']['result']['text'])==authored:matched=True
        if not matched:raise Invalid('accepted reply lacks completed journal binding')
    saved=read_json(root/'requests/04-0.json')
    payload=role_check.payload_for(old,'Goal',result['history'])
    messages=methods.Roles.messages(None,'Goal',payload)
    if saved['key']['plan']!=old['hash'] or saved['key']['payload']!=payload or saved['messages']!=messages:
        raise Invalid('missing Goal request differs from preserved evidence')
    counter=counter or role_tokens.measure
    measured=counter(old['agents'],messages)
    old_cap=old['agents']['local']['max_input_tokens'];new_cap=10240
    if not old_cap<measured['input_tokens']<=new_cap:raise Invalid('expected over-cap request must fit reviewed 10240-token recovery cap')
    agents=copy.deepcopy(old['agents']);agents['local']['max_input_tokens']=new_cap
    return c.seal(dict(derived,schema='seek-role-goal-recovery-v1',agents=agents,
        failed_root=str(root.resolve()),failed_plan_hash=old['hash'],failed_result_hash=result['hash'],
        initial_history=copy.deepcopy(result['history']),roles=['Goal'],max_logical_calls=1,max_attempts=1,attempts_per_call=1,
        old_input_cap=old_cap,new_input_cap=new_cap,input_preflight=measured,
        origin='evaluator_selected_development_recovery',scope='Complete only missing Goal review; no new experiment',
        recovery_messages_hash=digest(messages)))


def approve(draft, reviewer):
    if not reviewer.strip():raise Invalid('reviewer required')
    return c.seal(dict(draft,review=dict(draft_hash=draft['hash'],accepted=True,reviewer=reviewer,
        reason='Reviewed exact preserved Goal request, 10240-token input cap, one defender attempt and zero victim calls')))


def validate(plan, counter=None):
    c.current(plan)
    draft=prepare(plan['failed_root'],counter)
    review=plan.get('review',{})
    if review.get('accepted') is not True or review.get('draft_hash')!=draft['hash'] or not review.get('reviewer'):
        raise Invalid('explicit recovery budget review required')
    if approve(draft,review['reviewer'])!=plan:raise Invalid('recovery differs from reviewed preserved Goal request')
