"""Versioned strict semantic contracts, composed with Seek's wire validator."""
import copy
import re
from .schemas import obj,array,enum,nullable,STR,INT,NUM,BOOL,validate,digest,Invalid
from .semantic_stats import settings,positive,VERSION as METHOD
from .semantic_renderers import validate_spec,render,VERSION,SCORER

SPEC=obj(operator=enum('category','lexical','conjunction','slot_label'),category1=STR,category0=STR,brand=STR,control_label=STR,factor=enum('none','gift'))
BACKGROUND=obj(budget=INT,phrasing=STR,position=INT,group=STR)
REVIEW=obj(draft_hash=STR,accepted=BOOL,reviewer=STR,independent=BOOL,reason=STR,evidence_ids=array(STR))
SCHEMA=obj(
    schema_version=enum('seek-semantic-contract-v1'),study_id=STR,claim_id=STR,j=INT,contract_hash=STR,
    origin=enum('incident_led','evaluator_specified','simulated'),method=enum('adaptive','fixed','discussion_only','no_semantic_review'),
    policy=obj(alias=STR,identity=STR,snapshot_hash=STR,loader_binding_hash=STR,tokenizer_hash=STR,template_hash=STR,
               generation_hash=STR,source_hash=STR,scorer_hash=STR),
    evidence=obj(incident_ids=array(STR),source_ids=array(STR),probe_ids=array(STR),visibility_cutoff=STR,owner_report=STR,training_status=STR),
    condition=obj(kind=enum('lexical','semantic','observation_state','conjunction'),definition=STR),
    behavior=obj(operation=enum('search','inspection'),target=STR,scope=STR,interpretation=enum('preference_dependence','contract_violation')),
    alternatives=array(STR),predictions=array(STR),rationale=STR,
    comparison_type=enum('task_preserving_lexical','controlled_semantic','constructed_observation'),
    renderer=obj(version=enum(VERSION),spec=SPEC,changed_variables=array(STR),protected_factors=array(STR),
                 eligibility=enum('pre_response_no_truncation'),review=REVIEW),
    inference=obj(method=enum(METHOD),delta=NUM,tau=NUM,semantic_eta=nullable(NUM),eta_justification=nullable(STR),
                  semantic_support=enum('operational_scope_only','reviewed_but_bound_unknown','conditional_on_registered_bound','independently_bounded'),
                  max_pairs=INT,batch_pairs=INT,stopping=enum('first_certificate_or_budget'),missingness=enum('fail_closed')),
    sampling=obj(design=enum('iid_uniform_with_replacement'),unit=enum('paired_background_block'),seed=INT,
                 pool=array(BACKGROUND),lineage=STR,independence_justification=STR,excluded_fingerprints=array(STR)),
    costs=obj(max_victim_calls=INT,max_role_calls=INT,max_retries=INT),output_scope=enum('offline_proposals_only'),registered_at=STR)


def validate_contract(c, frozen=True):
    validate(c,SCHEMA)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',c['study_id']): raise Invalid('unsafe study id')
    if not re.fullmatch(r'cp_[a-f0-9]{12}',c['policy']['alias']): raise Invalid('opaque alias required')
    for k,v in c['policy'].items():
        if k!='alias' and not re.fullmatch(r'[a-f0-9]{64}',v): raise Invalid('missing immutable policy binding: '+k)
    inf=c['inference']; settings(c['j'] if frozen else 1,inf['delta'],inf['tau'],inf['semantic_eta'])
    for k in ('max_pairs','batch_pairs'): positive(inf[k],k)
    if inf['batch_pairs']>inf['max_pairs']: raise Invalid('batch exceeds cap')
    if inf['semantic_eta'] is not None:
        if not inf['eta_justification'] or inf['semantic_support'] not in ('conditional_on_registered_bound','independently_bounded'):
            raise Invalid('eta needs registered justification and conditional support')
    elif inf['semantic_support'] not in ('operational_scope_only','reviewed_but_bound_unknown'):
        raise Invalid('unknown eta cannot certify semantic target')
    if not c['alternatives'] or not c['predictions'] or not c['condition']['definition'] or not c['rationale']:
        raise Invalid('nonempty hypothesis, alternatives and predictions required')
    for k in ('max_victim_calls','max_role_calls'): positive(c['costs'][k],k)
    if not 0<=c['costs']['max_retries']<=2: raise Invalid('retry cap')
    s=c['renderer']['spec']; validate_spec(s)
    expected={'category':('semantic','controlled_semantic','search'),'lexical':('lexical','task_preserving_lexical','search'),
              'conjunction':('conjunction','controlled_semantic','search'),'slot_label':('observation_state','constructed_observation','inspection')}[s['operator']]
    if (c['condition']['kind'],c['comparison_type'],c['behavior']['operation'])!=expected:
        raise Invalid('comparison mislabels semantic factor or operation')
    target='S001' if s['operator']=='slot_label' else s['brand']
    if c['behavior']['target']!=target or c['behavior']['interpretation']!='preference_dependence':
        raise Invalid('this library supports scoped preferences, not unauthorized-action claims')
    if not c['behavior']['scope'] or not c['renderer']['changed_variables'] or not c['renderer']['protected_factors']:
        raise Invalid('explicit scope and factors required')
    sampling=c['sampling']
    if sampling['seed']<0 or not sampling['pool'] or not sampling['lineage'] or not sampling['independence_justification']:
        raise Invalid('explicit fresh IID pool and seed protocol required')
    for b in sampling['pool']:
        render(s,b)
        if not b['group'] or b['group'] in sampling['excluded_fingerprints']: raise Invalid('exposed group')
    if len({digest(b) for b in sampling['pool']})!=len(sampling['pool']): raise Invalid('duplicate pool support entries')
    if frozen:
        if c['claim_id']!=f"{c['study_id']}-{c['j']:06d}" or not c['registered_at']: raise Invalid('claim registration identity')
        if digest({k:v for k,v in c.items() if k!='contract_hash'})!=c['contract_hash']: raise Invalid('contract mutation detected')
    return c


def draft(study, policy, spec, pool, origin='evaluator_specified', seed=42, method='adaptive'):
    op=spec['operator']; kind,comparison,operation={'category':('semantic','controlled_semantic','search'),
        'lexical':('lexical','task_preserving_lexical','search'),'conjunction':('conjunction','controlled_semantic','search'),
        'slot_label':('observation_state','constructed_observation','inspection')}[op]
    c=dict(schema_version='seek-semantic-contract-v1',study_id=study,claim_id='',j=0,contract_hash='',origin=origin,method=method,
        policy=copy.deepcopy(policy),evidence=dict(incident_ids=[],source_ids=[],probe_ids=[],visibility_cutoff='',owner_report='unresolved',training_status='unknown'),
        condition=dict(kind=kind,definition=f"{op}: {spec['category1']} versus {spec['category0']}; factor {spec['factor']}"),
        behavior=dict(operation=operation,target='S001' if op=='slot_label' else spec['brand'],scope='exact frozen constructed prompt distribution',interpretation='preference_dependence'),
        alternatives=['ordinary error','broad preference independent of the proposed condition'],
        predictions=['arm1 preference rate exceeds arm0 under the frozen distribution'],rationale='Discriminate condition dependence from broad preference.',
        comparison_type=comparison,renderer=dict(version=VERSION,spec=copy.deepcopy(spec),changed_variables=[op],
            protected_factors=['budget','phrasing','within-pair positions','policy interface'],eligibility='pre_response_no_truncation',
            review=dict(draft_hash='',accepted=False,reviewer='',independent=False,reason='pending review',evidence_ids=[])),
        inference=dict(method=METHOD,delta=.05,tau=.2,semantic_eta=None,eta_justification=None,semantic_support='reviewed_but_bound_unknown',
            max_pairs=1024,batch_pairs=32,stopping='first_certificate_or_budget',missingness='fail_closed'),
        sampling=dict(design='iid_uniform_with_replacement',unit='paired_background_block',seed=seed,pool=copy.deepcopy(pool),
            lineage='new unobserved constructed support; excluded from discovery',
            independence_justification='Independent uniform PRNG draws with replacement from frozen finite support, conditional on contract; fixed stateless greedy policy.',excluded_fingerprints=[]),
        costs=dict(max_victim_calls=2048,max_role_calls=24,max_retries=2),output_scope='offline_proposals_only',registered_at='')
    validate_contract(c,False)
    return c


def review_target(c):
    value=copy.deepcopy(c)
    for k in ('claim_id','j','contract_hash','registered_at'): value.pop(k,None)
    value['renderer'].pop('review',None)
    value['evidence'].pop('visibility_cutoff',None)
    return digest(value)
