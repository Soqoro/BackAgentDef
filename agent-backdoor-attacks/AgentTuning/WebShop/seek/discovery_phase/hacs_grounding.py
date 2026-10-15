"""Post-inspection grounding diagnostic: no registry, victim, or search calls."""
import copy
from pathlib import Path
from ..schemas import Invalid
from ..storage import Journal, immutable_json, read_json
from . import common as c, hacs_runtime as r


def prepare(result_path):
    root=Path(result_path).parent
    result=c.checked(read_json(result_path));old=c.checked(read_json(root/'plan.json'))
    if result['plan_hash']!=old['hash']:raise Invalid('parent_plan_binding')
    if result['schema']!='seek-hacs-result-v1':raise Invalid('parent_schema')
    incident=copy.deepcopy(old['incidents'][result['index']])
    return c.seal(dict(schema='seek-hacs-grounding-development-v1',source=c.source(),
        parent_result_hash=result['hash'],parent_plan_hash=old['hash'],parent_j=result['j'],
        incident=incident,domain=old['domain'],agents=old['agents'],simulated=old['simulated'],
        input_cap=min(4096,old['input_cap']),defender_budget=1,maximum_physical_calls=2,
        victim_calls=0,registry_mutations=0,development_reuse=True,
        method_matrix_unlocked=False,scientific_confirmation=False,
        scope='post-inspection Action grounding only; no method comparison or fresh discovery'))


def validate(plan):
    c.current(plan)
    if plan['schema']!='seek-hacs-grounding-development-v1':raise Invalid('development_schema')
    if (plan['defender_budget'],plan['maximum_physical_calls'],plan['victim_calls'])!=(1,2,0):raise Invalid('development_caps')
    if plan['input_cap']>4096 or not plan['development_reuse'] or plan['method_matrix_unlocked']:raise Invalid('development_scope')


def packet(plan):
    return r.packet(plan,plan['incident'],phase='grounding')


def run(plan,backend,out):
    validate(plan)
    if backend.simulated!=plan['simulated']:raise Invalid('population_mismatch')
    out=Path(out)
    if (out/'result.json').exists():
        result=c.checked(read_json(out/'result.json'))
        if result['plan_hash']!=plan['hash']:raise Invalid('development_plan_binding')
        return result
    if (out/'started.json').exists():raise Invalid('interrupted_development_requires_review')
    immutable_json(out/'plan.json',plan)
    immutable_json(out/'started.json',dict(plan_hash=plan['hash']))
    p=packet(plan);immutable_json(out/'packet.json',p)
    journal=Journal(out/'defender');review=None;failure=None
    try:
        review=r.review_call(plan,backend,'Action',p,out,journal,1)
        immutable_json(out/'review.json',review)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc))
    finally:backend.close()
    result=c.seal(dict(schema='seek-hacs-grounding-development-result-v1',plan_hash=plan['hash'],
        parent_j=plan['parent_j'],parent_result_hash=plan['parent_result_hash'],
        status='backend_failure' if failure else 'abstained' if review['decision']=='abstain' else 'review_completed',
        review=review,failure=failure,defender_costs=r.token_cost(out,'defender'),
        simulated=plan['simulated'],development_reuse=True,victim_calls=0,registry_mutations=0,
        method_matrix_unlocked=False,scientific_confirmation=False,scope=plan['scope']))
    immutable_json(out/'result.json',result)
    return result
