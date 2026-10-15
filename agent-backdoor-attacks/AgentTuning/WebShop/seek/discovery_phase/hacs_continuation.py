"""Bounded development probes for an already agent-selected output target."""
import copy
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import Journal, immutable_json, read_json
from . import common as c, hacs as h, hacs_runtime as r


def prepare(result_path, parent_plan_path):
    result=c.checked(read_json(result_path))
    grounding=c.checked(read_json(Path(result_path).parent/'plan.json'))
    parent=c.checked(read_json(parent_plan_path))
    if result['schema']!='seek-hacs-grounding-development-result-v1' or result['failure'] or result['review']['decision']!='continue':
        raise Invalid('completed_agent_grounding_required')
    if result['plan_hash']!=grounding['hash'] or grounding['parent_plan_hash']!=parent['hash']:
        raise Invalid('development_parent_binding')
    target=result['review']['target'];incident=grounding['incident']
    spec=h.outcome(incident['public'],target)
    h.check_outcome_compatibility(target,grounding['domain']['forms'])
    return c.seal(dict(schema='seek-hacs-continuation-development-v1',source=c.source(),
        grounding_result_hash=result['hash'],parent_j=result['parent_j'],parent_plan_hash=parent['hash'],
        domain=grounding['domain'],incident=incident,agents=grounding['agents'],profile=parent['profile'],
        outcome=spec,simulated=grounding['simulated'],development_reuse=True,
        input_cap=4096,defender_budget=4,maximum_defender_attempts=8,victim_budget=4,anchors=2,
        context_start=5000,registry_mutations=0,method_matrix_unlocked=False,scientific_confirmation=False,
        real_victim_execution='exit_scoped_subprocess_v1',
        scope='post-inspection two-pair controller development; no fresh discovery or effect certificate'))


def validate(plan):
    c.current(plan)
    if plan['schema']!='seek-hacs-continuation-development-v1':raise Invalid('continuation_schema')
    if (plan['input_cap'],plan['defender_budget'],plan['maximum_defender_attempts'],plan['victim_budget'],plan['anchors'],plan['context_start'])!=(4096,4,8,4,2,5000):
        raise Invalid('continuation_caps')
    if not plan['development_reuse'] or plan['method_matrix_unlocked']:raise Invalid('continuation_scope')
    if plan.get('real_victim_execution')!='exit_scoped_subprocess_v1':raise Invalid('continuation_worker_contract')
    if h.outcome(plan['incident']['public'],plan['outcome']['target'])!=plan['outcome']:raise Invalid('outcome_binding')


def search_for(plan):
    family=h.hypotheses(plan['domain'],plan['outcome']['id'])
    s=h.Search(plan['domain'],family,h.bank(plan['domain'],family));s.outcome_spec=plan['outcome']
    return s


def packet(plan, search):
    p=r.packet(plan,plan['incident'],search,phase='development_evidence_review')
    # Review needs realized evidence and a few unperformed alternatives, not the
    # entire grammar. Selection itself still uses the full bank and rule family.
    keep=set(search.used)|set(p['unperformed'][:4])
    original=len(p['available']);p['available']=[e for e in p['available'] if e['id'] in keep]
    p['unperformed']=[x for x in p['unperformed'] if x in keep]
    p['review_display']=dict(omitted_experiments=original-len(p['available']),selection_bank_unchanged=True)
    p['development_reuse']=True
    p['scope']=plan['scope'];return p


def run(plan,backend,loader,out):
    validate(plan)
    if backend.simulated!=plan['simulated']:raise Invalid('population_mismatch')
    out=Path(out)
    if (out/'result.json').exists():
        result=c.checked(read_json(out/'result.json'))
        if result['plan_hash']!=plan['hash']:raise Invalid('continuation_binding')
        return result
    if (out/'started.json').exists():raise Invalid('interrupted_continuation_requires_review')
    immutable_json(out/'plan.json',plan);immutable_json(out/'started.json',dict(plan_hash=plan['hash']))
    search=search_for(plan);journal=Journal(out/'defender');rows=[];reviews=[];anchors=None
    failure=None;stable=None;status='development_complete';logical=0
    def review(role):
        nonlocal logical
        logical+=1
        p=packet(plan,search);immutable_json(out/'packets'/f'{logical:02d}.json',p)
        value=r.review_call(plan,backend,role,p,out,journal,logical,plan['outcome']['target'])
        reviews.append(value);immutable_json(out/'reviews'/f'{logical:02d}.json',value)
        return value
    def batch(cells,stage):
        backend.close()
        if not plan['simulated']:
            from .hacs_victim_worker import run_batch
            return run_batch(plan,cells,out,stage)
        # Existing CPU fake backends need no CUDA process isolation.
        victim=loader(plan['profile']['models']['query'],plan['profile']['generation'])
        try:
            if victim.simulated!=plan['simulated']:raise Invalid('victim_population_mismatch')
            return c.responses(victim,cells,plan['profile']['generation'],out/'victim',plan['hash'],h.score)
        finally:c.release(victim)
    try:
        for i in range(2):
            decision=search.choose('hacs_active',4-2*i)
            immutable_json(out/'decisions'/f'{i:03d}.json',decision)
            if decision['experiment'] is None:status=decision['status'];break
            experiment=next(e for e in search.experiments if e['id']==decision['experiment'])
            cells=h.render(experiment,plan['outcome']['target'],5000+i)
            if anchors is None:anchors=copy.deepcopy(cells);immutable_json(out/'anchor_plan.json',anchors)
            responses=batch(cells,f'pair-{i:03d}')
            rows.extend(responses)
            observations=[dict(value=x['score']['value'],raw_response=x['reply']['raw_response'],action=x['score']['action'],
                prompt=x['reply']['serialized_prompt'],tokens=x['reply']['encoded_ids'],runtime=x['runtime'],
                physical_key=x['key'],source_group=plan['incident']['group'],cell=x['cell']) for x in responses]
            event=search.update(decision,observations);immutable_json(out/'evidence'/f'{i:03d}.json',event)
            if any(x['score']['value'] is None for x in responses):status='unscorable_probe';break
            value=review('Action')
            if value['decision'] in ('abstain','stop','finalize'):status='review_stopped';break
        if rows:
            repeats=batch([dict(x,id='anchor-'+x['id'],anchor=True) for x in anchors],'anchors')
            stable=all(a['reply']['raw_response']==b['reply']['raw_response'] for a,b in zip(rows[:2],repeats))
            if not stable:status='determinism_unresolved'
        if status=='development_complete' and len(rows)==4:
            for role in ('Goal','State'):
                if review(role)['decision'] in ('abstain','stop'):status='review_stopped';break
    except Exception as exc:failure=dict(type=type(exc).__name__,reason=str(exc));status='backend_failure'
    finally:
        backend.close()
    result=c.seal(dict(schema='seek-hacs-continuation-development-result-v1',plan_hash=plan['hash'],parent_j=plan['parent_j'],
        status=status,failure=failure,simulated=plan['simulated'],development_reuse=True,outcome=plan['outcome'],
        completed_pairs=len(rows)//2,search_facts=search.facts(),reviews=reviews,anchors_stable=stable,
        defender_costs=r.token_cost(out,'defender'),victim_costs=r.token_cost(out/'victim'),
        registry_mutations=0,method_matrix_unlocked=False,scientific_confirmation=False,sampling_certificate=False,
        exposure_scope='development consumed prompts retained locally; not reserved in authoritative registry; never reuse as held-out evidence',
        scope=plan['scope']))
    immutable_json(out/'result.json',result);return result
