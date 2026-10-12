"""Reviewed development probe of an unchanged role-authored request."""
import copy
import json
from pathlib import Path
from ..schemas import Invalid
from ..storage import immutable_json
from . import common as c, methods, panels

LIMITATIONS = [
    'Post-inspection development reuse of one selected incident; not a method comparison.',
    'Constructed next-action prompts, not executed purchases or episode rewards.',
    'Probe frames omit original incident attributes and prices; this is not a task-preserving replay.',
    'Agent condition labels do not establish semantic validity or malicious training.',
    'Similar footwear terms do not remove the sneaker category.',
    'Equal arm rates would weaken wording sensitivity on these prompts; they do not rule out other effects.',
    'A nonzero descriptive difference does not establish semantic activation or backdoor causation.',
    'No held-out evaluation, sampling certificate, or automatic full-matrix approval.'
]


def prepare(parent, role_plan, result):
    parent,role_plan,result=json.loads(json.dumps([parent,role_plan,result]))
    for artifact in (parent, role_plan, result): c.checked(artifact)
    if parent['schema']!='seek-query-method-v2' or role_plan['schema']!='seek-role-grounding-check-v1' or result['schema']!='seek-role-grounding-result-v1':
        raise Invalid('method parent and role-check artifacts required')
    if role_plan['source_plan_hash']!=parent['hash'] or result['plan_hash']!=role_plan['hash'] or result['source_plan_hash']!=parent['hash']:
        raise Invalid('role source binding mismatch')
    index=role_plan['source_incident_index']
    if type(index) is not int or not 0<=index<4 or result['source_incident_index']!=index:
        raise Invalid('incident binding mismatch')
    if result['status']!='completed_role_check' or result['failure_details'] or result['victim_calls']!=0 or result['registry_mutations']!=0:
        raise Invalid('completed no-victim role check required')
    if any(x['simulated']!=parent['simulated'] for x in (role_plan,result)):
        raise Invalid('population mismatch')
    bank=methods.compile_bank(dict(status='ready',selected=parent['incidents']),parent['profile'],simulated=parent['simulated'])
    if role_plan['incident']!=bank['incidents'][index]:raise Invalid('role incident differs from bound source')
    history=[]
    if len(result['history'])!=4:raise Invalid('four role replies required')
    for role,h in zip(('Action','Goal','State','Action'),result['history']):
        authored={k:v for k,v in h.items() if k not in ('compilation','experiment_status')}
        check=methods.compile_hypothesis(authored,role,history,role_plan['incident']['public'],[])
        if check['status']!='executable' or h['compilation']!=check or h['experiment_status']!='executable':
            raise Invalid('role request is not executable under current compiler')
        history.append(authored)
    candidate=history[2];approval=history[3]
    if not approval['accepted'] or candidate['request']!=approval['request']:raise Invalid('final approval changed or rejected candidate')
    cells=methods.cells(candidate['request'],'discovery',role_plan['incident']['group'],n=4)
    return c.seal(dict(schema='seek-role-development-probe-v1',source=c.source(),
        parent=copy.deepcopy(parent),role_plan=copy.deepcopy(role_plan),role_result=copy.deepcopy(result),
        profile=dict(models={'query':copy.deepcopy(parent['profile']['models']['query'])},generation=copy.deepcopy(parent['profile']['generation'])),
        candidate=copy.deepcopy(candidate),cells=cells,anchor_ids=[cells[0]['id'],cells[1]['id']],
        simulated=parent['simulated'],origin='agent_authored_contrast_evaluator_selected_development_probe',
        development_reuse=True,core_responses=8,maximum_physical_responses=10,defender_calls=0,
        claim_allocations=0,registry_policy='append consumed-input exposure records only; preserve existing claims',
        limitations=LIMITATIONS,scientific_confirmation=False,sampling_certificate=False,semantic_eta=None))


def validate(plan):
    c.current(plan)
    if prepare(plan['parent'],plan['role_plan'],plan['role_result'])!=plan:raise Invalid('probe differs from derived authored request or budget')


def check_review(plan, review):
    validate(plan)
    if review.get('plan_hash')!=plan['hash'] or review.get('accepted') is not True or not review.get('reviewer') or not review.get('reason'):
        raise Invalid('explicit input, scope and ten-response budget review required')


def run(plan, review, loader, reg, output):
    check_review(plan,review)
    out=Path(output)
    immutable_json(out/'plan.json',plan);immutable_json(out/'review.json',review)
    if not (out/'query/result.json').exists() and c.costs(out/'query')['physical_calls']:
        raise Invalid('interrupted probe with prior attempts; preserve output and review, no automatic retry')
    def bound_loader(entry,generation):
        v=loader(entry,generation)
        if v.identity!=entry['identity']:
            c.release(v);raise Invalid('victim identity differs from frozen profile')
        return v
    result=panels.run(plan,'query',bound_loader,reg,out)
    summary=dict(schema='seek-role-development-probe-result-v1',plan_hash=plan['hash'],
        status=result['status'],simulated=plan['simulated'],development_reuse=True,
        origin=plan['origin'],request=plan['candidate']['request'],costs=result['costs'],
        defender_calls=0,claim_allocations=0,scientific_confirmation=False,sampling_certificate=False,
        limitations=plan['limitations'],paired_results=[],descriptive_mean=None)
    if 'rows' in result:
        c.verify_rows(result,plan,out/'query',panels.score)
        summary['anchors_stable']=result['anchors_stable']
        rows={r['cell']['id']:r for r in result['rows']}
        for i in range(4):
            left=rows[f'{i:03d}-1']['score']['value'];right=rows[f'{i:03d}-0']['score']['value']
            summary['paired_results'].append(dict(pair=i,left=left,right=right,difference=None if left is None or right is None else left-right))
        if result['status']=='complete':summary['descriptive_mean']=sum(p['difference'] for p in summary['paired_results'])/4
    else:summary['error']=result.get('error')
    summary=c.seal(summary);immutable_json(out/'result.json',summary)
    return summary
