"""Bounded development-reuse role check; no victim loader or study allocation."""
import copy
import json
import subprocess
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import Journal, immutable_json, read_json
from . import common as c, methods


def payload_for(plan, role, history):
    payload=dict(incident=copy.deepcopy(plan['incident']['public']),history=copy.deepcopy(history),
        evidence=copy.deepcopy(plan.get('evidence',[])),role=role,library=methods.library(),round=0,
        phase='development_role_check',remaining_victim=0,must_finalize=True)
    if 'followup_task' in plan:payload.update(followup_task=plan['followup_task'],evidence_limits=plan['evidence_limits'])
    return payload


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
    recovery=plan.get('schema')=='seek-role-goal-recovery-v1'
    followup=recovery or plan.get('schema')=='seek-role-evidence-followup-v1'
    if recovery:
        from .role_recovery import validate as validate_recovery
        validate_recovery(plan)
    elif followup:
        from .role_followup import validate as validate_followup
        validate_followup(plan)
    else:validate(plan)
    if bool(backend.simulated) != plan['simulated']:
        raise Invalid('backend population mismatch')
    out=Path(output)
    if (out/'result.json').exists():
        result=c.checked(read_json(out/'result.json'))
        if result['plan_hash']!=plan['hash']:raise Invalid('role check resume mismatch')
        return result
    immutable_json(out/'plan.json',plan)
    immutable_json(out/'public.json',plan['incident']['public'])
    journal=Journal(out/'defender');history=copy.deepcopy(plan['initial_history']) if followup else [];failures=[];logical=0;attempts=0;failed=False
    evidence=copy.deepcopy(plan['evidence']) if followup else []
    try:
        for role in plan['roles']:
            logical+=1;accepted=None
            payload=payload_for(plan,role,history)
            for retry in range(plan['attempts_per_call']):
                reply=None;stage='preflight'
                key=dict(plan=plan['hash'],role=role,call=logical,retry=retry,payload=copy.deepcopy(payload))
                kh=digest(key)
                if journal.completed(kh) is None and any(e['kind']=='call_attempt' and e['data']['key']==kh for e in journal.records):
                    raise Invalid('unknown prior role execution; preserve and inspect interrupted output')
                immutable_json(out/'requests'/f'{logical:02d}-{retry}.json',dict(key=key,messages=backend.messages(role,payload)))
                try:
                    if not backend.simulated:
                        from .role_tokens import measure
                        counts=measure(plan['agents'],backend.messages(role,payload))
                    elif hasattr(backend,'preflight'):counts=backend.preflight(backend.messages(role,payload))
                    else:counts=dict(input_tokens=None,simulated=True)
                    immutable_json(out/'preflight'/f'{logical:02d}-{retry}.json',counts)
                    cap=plan['agents'].get('local',{}).get('max_input_tokens')
                    if counts['input_tokens'] is not None and cap is not None and counts['input_tokens']>cap:
                        raise Invalid('defender input exceeds cap; truncation forbidden')
                    stage='transport';attempts+=1
                    reply=journal.call(key,'defender','development_role_check',plan['max_attempts'],lambda:backend.call(role,payload))
                    stage='validation'
                    if reply.get('finish_reason')!='stop' or reply.get('refusal'):raise Invalid('incomplete reply')
                    value=json.loads(reply['text'])
                    check=methods.compile_hypothesis(value,role,history,payload['incident'],[e['id'] for e in evidence])
                    if followup and not value['probe_ids']:raise Invalid('followup must reference supplied probe evidence')
                    accepted=dict(value,compilation=check,experiment_status=check['status'])
                    history.append(accepted);break
                except (Invalid,ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as exc:
                    over_cap=isinstance(exc,Invalid) and 'defender input exceeds cap; truncation forbidden' in str(exc)
                    code='input_cap_exceeded' if over_cap else str(exc) if stage=='validation' and isinstance(exc,Invalid) else type(exc).__name__
                    detail=dict(role=role,call=logical,retry=retry,stage=stage,code=code)
                    failures.append(detail);journal.emit('role_failure',detail)
                    if over_cap or stage=='preflight':break
                    payload.update(validation_error=code,previous_reply=reply.get('text','') if isinstance(reply,dict) else '')
            if accepted is None:failed=True;break
    finally:
        backend.close()
    result=c.seal(dict(schema='seek-role-evidence-followup-result-v1' if followup else 'seek-role-grounding-result-v1',plan_hash=plan['hash'],
        status='backend_failure' if failed else 'completed_role_check',simulated=plan['simulated'],
        origin=plan['origin'],development_reuse=True,source_plan_hash=plan['source_plan_hash'],
        source_incident_index=plan['source_incident_index'],history=history,failure_details=failures,
        defender_logical_calls=logical,defender_attempts=attempts,defender_costs=c.costs(out/'defender',journal_subdir=''),
        victim_calls=0,registry_mutations=0,scientific_confirmation=False,sampling_certificate=False,
        semantic_interpretation='requires human review',
        scope='Role interpretation only; no probes, candidate evaluation, or method comparison'))
    if followup:result=c.seal(dict(result,initial_history_entries=len(plan['initial_history']),new_history_entries=len(history)-len(plan['initial_history']),probe_result_hash=plan['probe_result_hash']))
    if recovery:result=c.seal(dict(result,schema='seek-role-goal-recovery-result-v1',failed_result_hash=plan['failed_result_hash']))
    immutable_json(out/'result.json',result)
    return result
