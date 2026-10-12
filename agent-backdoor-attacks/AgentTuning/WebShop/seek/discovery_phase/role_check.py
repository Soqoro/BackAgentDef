"""Bounded development-reuse role check; no victim loader or study allocation."""
import copy
import json
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import Journal, immutable_json, read_json
from . import common as c, methods


def prepare(parent, index, agents):
    c.checked(parent)  # Read historical evidence without claiming current source identity.
    if parent['schema'] != 'seek-query-method-v2' or type(index) is not int or not 0 <= index < 4:
        raise Invalid('existing method plan and incident index 0..3 required')
    if not parent['simulated']:
        from ..local_roles import validate_local
        validate_local(agents)
        if type(agents.get('max_output_tokens')) is not int or not 1 <= agents['max_output_tokens'] <= 1024:
            raise Invalid('bounded role output cap required')
    bank = methods.compile_bank(dict(status='ready', selected=parent['incidents']),
                                parent['profile'], simulated=parent['simulated'])
    return c.seal(dict(schema='seek-role-grounding-check-v1', source=c.source(),
        simulated=parent['simulated'], development_reuse=True,
        origin='evaluator_selected_engineering_case',
        selection_reason='Previously inspected incident selected to diagnose role interpretation; not a fresh method trial',
        scope='development role interpretation check',
        source_allocation_j=parent['j'], source_plan_hash=parent['hash'], source_incident_index=index,
        incident=bank['incidents'][index], agents=copy.deepcopy(agents),
        roles=['Action','Goal','State','Action'], max_logical_calls=4, max_attempts=8,
        attempts_per_call=2, victim_calls=0, registry_mutations=0,
        scientific_confirmation=False, semantic_interpretation='requires human review'))


def validate(plan):
    c.current(plan)
    if plan['schema'] != 'seek-role-grounding-check-v1' or plan['roles'] != ['Action','Goal','State','Action']:
        raise Invalid('role check schema/order mismatch')
    if (plan['max_logical_calls'],plan['max_attempts'],plan['attempts_per_call'],plan['victim_calls']) != (4,8,2,0):
        raise Invalid('role check budget changed')
    if plan['development_reuse'] is not True or plan['scientific_confirmation'] is not False:
        raise Invalid('development-only scope required')


def run(plan, backend, output):
    validate(plan)
    if bool(backend.simulated) != plan['simulated']:
        raise Invalid('backend population mismatch')
    out=Path(output)
    if (out/'result.json').exists():
        result=c.checked(read_json(out/'result.json'))
        if result['plan_hash']!=plan['hash']:raise Invalid('role check resume mismatch')
        return result
    immutable_json(out/'plan.json',plan)
    immutable_json(out/'public.json',plan['incident']['public'])
    journal=Journal(out/'defender');history=[];failures=[];logical=0;attempts=0;failed=False
    try:
        for role in plan['roles']:
            logical+=1;accepted=None
            payload=dict(incident=copy.deepcopy(plan['incident']['public']),history=copy.deepcopy(history),
                         evidence=[],role=role,library=methods.library(),round=0,
                         phase='development_role_check',remaining_victim=0,must_finalize=True)
            for retry in range(2):
                reply=None;stage='transport';attempts+=1
                key=dict(plan=plan['hash'],role=role,call=logical,retry=retry,payload=copy.deepcopy(payload))
                kh=digest(key)
                if journal.completed(kh) is None and any(e['kind']=='call_attempt' and e['data']['key']==kh for e in journal.records):
                    raise Invalid('unknown prior role execution; preserve and inspect interrupted output')
                immutable_json(out/'requests'/f'{logical:02d}-{retry}.json',dict(key=key,messages=backend.messages(role,payload)))
                try:
                    reply=journal.call(key,'defender','development_role_check',8,lambda:backend.call(role,payload))
                    stage='validation'
                    if reply.get('finish_reason')!='stop' or reply.get('refusal'):raise Invalid('incomplete reply')
                    value=json.loads(reply['text'])
                    check=methods.compile_hypothesis(value,role,history,payload['incident'],[])
                    accepted=dict(value,compilation=check,experiment_status=check['status'])
                    history.append(accepted);break
                except (Invalid,ValueError,KeyError,TypeError,OSError) as exc:
                    code=str(exc) if stage=='validation' and isinstance(exc,Invalid) else type(exc).__name__
                    detail=dict(role=role,call=logical,retry=retry,stage=stage,code=code)
                    failures.append(detail);journal.emit('role_failure',detail)
                    payload.update(validation_error=code,previous_reply=reply.get('text','') if isinstance(reply,dict) else '')
            if accepted is None:failed=True;break
    finally:
        backend.close()
    result=c.seal(dict(schema='seek-role-grounding-result-v1',plan_hash=plan['hash'],
        status='backend_failure' if failed else 'completed_role_check',simulated=plan['simulated'],
        origin=plan['origin'],development_reuse=True,source_plan_hash=plan['source_plan_hash'],
        source_incident_index=plan['source_incident_index'],history=history,failure_details=failures,
        defender_logical_calls=logical,defender_attempts=attempts,defender_costs=c.costs(out/'defender',journal_subdir=''),
        victim_calls=0,registry_mutations=0,scientific_confirmation=False,sampling_certificate=False,
        semantic_interpretation='requires human review',
        scope='Role interpretation only; no probes, candidate evaluation, or method comparison'))
    immutable_json(out/'result.json',result)
    return result
