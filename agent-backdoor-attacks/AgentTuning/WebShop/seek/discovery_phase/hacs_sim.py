"""Known-rule HACS tests; no checkpoint recovery or soft-search theorem claim."""
import copy
import math
import random
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import immutable_json, read_json, row_lock
from ..next_phase.simulation import wilson
from . import hacs as h, common as c, campaign

CASES=('literal','concept','parent','constant_high','constant_low','conjunction','position','fixed_id',
       'missing_true_rule','equivalent','unsupported','noisy','malformed','biased_screening','stale_cache','unknown_mapping')


def fixture():
    return dict(schema='seek-hacs-domain-v1',version=1,provenance='fictional tool metadata, simulated declared prior',
        nodes=['fine','heavy','tools','wear'],forms={'fine tools':'fine','precision tools':'fine','heavy tools':'heavy','garments':'wear'},
        edges=[dict(child=x,parent='tools',type='is_a',provenance='fictional taxonomy',review='declared_prior') for x in ('fine','heavy')])


def trace(case,variant,seed=1,cap=16):
    rng=random.Random(seed);domain=fixture()
    if case=='unknown_mapping':domain['forms']['mystery tools']=None
    family=h.hypotheses(domain,'fixture-event',64,True)
    bank=h.bank(domain,family)
    if case=='unsupported':bank=bank[:1]
    # Observation controls expose factual fields; use same AST/update/selection,
    # not an oracle shortcut selecting a desired experiment.
    if case in ('position','fixed_id','conjunction'):
        for i,e in enumerate(bank):
            for a,f in enumerate(e['features']):
                f.update(target_position=1+(i+a)%2,target_id='observed_id' if (i+a)%2 else 'other',target_attribute='requested',source='observation',history_window='current_only')
            e['predictions']={x['id']:[None if (v:=h.evaluate(x['ast'],f,domain)) is None else .9 if v else .1 for f in e['features']] for x in family['hypotheses']}
    ast={'literal':dict(op='surface_form',value='fine tools'),'concept':dict(op='category_concept',value='fine'),
         'parent':dict(op='category_concept',value='tools'),'constant_high':dict(op='constant',value=True),
         'constant_low':dict(op='constant',value=False),'position':dict(op='position',value=1),
         'fixed_id':dict(op='fixed_id',value='observed_id'),
         'conjunction':dict(op='and',args=[dict(op='category_concept',value='fine'),dict(op='position',value=1)])}.get(case,dict(op='category_concept',value='fine'))
    search=h.Search(domain,family,bank);logical=0
    for i in range(cap//2):
        requested=bank[i%len(bank)]['id'] if variant=='role_selected' else None
        dec=search.choose(variant,cap-logical,requested)
        if dec['experiment'] is None:break
        e=next(x for x in bank if x['id']==dec['experiment']);obs=[]
        for a,f in enumerate(e['features']):
            y=h.evaluate(ast,f,domain)
            if case=='missing_true_rule':y=bool((i+a)%2)
            if case=='noisy' and rng.random()<.2:y=not y
            if case=='malformed' and rng.random()<.25:y=None
            obs.append(dict(value=None if y is None else int(y),prompt=str(f),tokens=list(str(f).encode()),runtime={'simulated':True}))
        search.update(dec,obs);logical+=2
        if case=='stale_cache':
            # A renamed reinspection must not update twice.
            before=search.weights();search.pending=dec;search.update(dec,obs)
            if before!=search.weights():raise Invalid('duplicate_weight_update')
    truth_sig=tuple(None if (v:=h.evaluate(ast,f,domain)) is None else .9 if v else .1 for e in bank for f in e['features'])
    true_ids=[x['id'] for x in family['hypotheses'] if tuple(p for e in bank for p in e['predictions'][x['id']])==truth_sig]
    if case=='missing_true_rule':true_ids=[]
    groups=search.coverage['classes'];weights=search.weights()
    best=max(groups,key=lambda g:sum(weights[x] for x in g)) if groups else []
    return dict(case=case,variant=variant,simulated=True,logical_responses=logical,truth_represented=bool(true_ids),
        ranked_equivalence_class_correct=bool(set(best)&set(true_ids)),predicted_class=best,
        unique_rule_recovery_claimed=False,false_confident_localization=False,
        nonidentifiable=bool(search.coverage['unresolved'] or search.coverage['unknown_hypotheses']),
        cap_censored=logical==cap,metrics=search.metrics,
        screening='biased positive incident sample deliberately not used as prevalence or prior' if case=='biased_screening' else 'neutral prior',
        semantic_certificate=False)


def elimination(predictions,true_rule,beta=.25,noise=0.,alpha=.05,seed=1):
    """Separate idealized backend; no guarantee transferred to soft HACS."""
    if true_rule not in predictions:return dict(status='assumption_failed_true_rule_missing',guarantee_applicable=False)
    if not 0<beta<=.5 or not 0<=noise<.5 or not 0<alpha<1:raise Invalid('elimination_assumptions')
    keys=set(next(iter(predictions.values())))
    if any(set(v)!=keys or any(x not in (0,1) for x in v.values()) for v in predictions.values()):raise Invalid('binary_aligned_bank')
    live=set(predictions);rng=random.Random(seed);trace=[];n=len(live)
    rounds=max(1,math.ceil(math.log(max(2,n))/-math.log(1-beta)))
    k=1 if noise==0 else math.ceil(math.log(rounds/alpha)/(2*(.5-noise)**2))
    if k%2==0:k+=1
    # Verify the balanced-split premise for every version space reachable by
    # this declared deterministic test-selection backend, not just the truth path.
    def check_spaces(space):
        if len(space)<=1:return True
        options=[]
        for test in sorted(keys):
            one={x for x in space if predictions[x][test]};zero=space-one
            if one and zero:options.append((max(len(one),len(zero)),test,one,zero))
        if not options:return False
        worst,_,one,zero=min(options,key=lambda x:(x[0],x[1]))
        return worst<=(1-beta)*len(space) and check_spaces(one) and check_spaces(zero)
    all_reachable_balanced=check_spaces(live)
    balanced=True
    while len(live)>1:
        choices=[]
        for t in sorted(keys):
            one={x for x in live if predictions[x][t]};zero=live-one
            if one and zero:choices.append((max(len(one),len(zero)),t,one,zero))
        if not choices:return dict(status='equivalence_class',rules=sorted(live),guarantee_applicable=False,trace=trace)
        worst,t,one,zero=min(choices,key=lambda x:(x[0],x[1]))
        balanced &= worst<=(1-beta)*len(live)
        votes=[predictions[true_rule][t] ^ int(rng.random()<noise) for _ in range(k)]
        label=int(sum(votes)>k/2);live=one if label else zero
        trace.append(dict(test=t,label=label,responses=k))
    return dict(status='identified_in_known_rule_fixture',rules=sorted(live),guarantee_applicable=all_reachable_balanced,
                all_reachable_spaces_checked=True,balanced_on_path=balanced,noise_independent_by_construction=True,noise_bound=noise,
                response_bound=rounds*k if balanced else None,trace=trace)


def run(output,replications=2,cap=16,seed=131013,large=False):
    if type(replications) is not int or not 1<=replications<=10000 or cap not in (16,32):raise Invalid('simulation_parameters')
    if replications>4 and not large:raise Invalid('full_campaign_requires_explicit_large')
    out=Path(output)
    if 'simulated' not in out.parts:raise Invalid('simulated_directory_required')
    manifest=c.seal(dict(schema='seek-hacs-simulation-v1',source=c.source(),simulated=True,seed=seed,replications=replications,cap=cap,cases=CASES,variants=h.VARIANTS))
    studies=[]
    with row_lock(out):
        immutable_json(out/'manifest.json',manifest)
        for i in range(replications):
            path=out/'studies'/f'{i:05d}.json'
            if path.exists():v=c.checked(read_json(path))
            else:
                rows=[trace(case,variant,seed+i*100003+k,cap) for k,case in enumerate(CASES) for variant in h.VARIANTS]
                # Existing independent confirmation draws; search scores never
                # enter certificates. Small horizon is explicitly cap-censored.
                cert=campaign.study(seed+i*100003,64,'fixed')
                v=c.seal(dict(manifest_hash=manifest['hash'],index=i,rows=rows,confirmation=cert));immutable_json(path,v)
            if v['manifest_hash']!=manifest['hash']:raise Invalid('simulation_resume_mismatch')
            studies.append(v)
        summaries=[]
        for case in CASES:
            for variant in h.VARIANTS:
                rs=[r for s in studies for r in s['rows'] if (r['case'],r['variant'])==(case,variant)]
                wins=sum(r['ranked_equivalence_class_correct'] for r in rs)
                summaries.append(dict(case=case,variant=variant,ranked_class_correct=wins,studies=len(rs),interval=wilson(wins,len(rs)),
                    cap_censored=sum(r['cap_censored'] for r in rs),logical_responses=sum(r['logical_responses'] for r in rs),
                    truth_represented=sum(r['truth_represented'] for r in rs)))
        false=sum(s['confirmation']['any_false_certificate'] for s in studies)
        positive=[a for s in studies for a in s['confirmation']['allocations'] if a['truth']>.2]
        result=c.seal(dict(schema='seek-hacs-simulation-result-v1',simulated=True,manifest_hash=manifest['hash'],completed_studies=len(studies),
            status='full_requested_campaign_completed' if replications>=1000 else 'smoke_only',discovery=summaries,
            certificate_family_false=dict(count=false,studies=replications,interval=wilson(false,replications)),
            positive_certificates=sum(a['certified'] for a in positive),positive_cases=len(positive),
            confirmation_horizon=64,confirmation_cap_censored=sum(a['blocks']==64 and not a['certified'] for s in studies for a in s['confirmation']['allocations']),
            logical_draws=sum(s['confirmation']['logical_responses']+sum(r['logical_responses'] for r in s['rows']) for s in studies),
            limitations=['Search rankings are not confidence statements','No trained-checkpoint recovery','Scripted-role baseline is not an LLM',
                        'Missing truth invalidates identification, not effect arithmetic','64-block certificate smoke can be underpowered; use existing full campaign for power']))
        immutable_json(out/'result.json',result);return result
