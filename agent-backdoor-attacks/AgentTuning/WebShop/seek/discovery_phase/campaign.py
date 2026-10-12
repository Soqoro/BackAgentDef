"""Resumable independent simulated studies using existing rule/radius primitives."""
import math
import random
import time
from pathlib import Path
from ..schemas import Invalid,digest
from ..storage import immutable_json,read_json,row_lock,atomic_json
from ..next_phase import design as d,simulation as old
from . import common as c

CASES=('null','boundary','strong','lexical_only','synonym_invariant','sparse_conjunction','missing_hypothesis','malformed','ordinary_preference','eta_misspecified')


def study(seed,cap,schedule):
    rng=random.Random(seed);coefs=[{'1':1,'0':-1},d.coefficients('observation','T'),d.coefficients('observation','T','R')]
    order=list(CASES);exploration=0
    if schedule=='adaptive':
        signals={}
        for case in order:
            truth={'strong':1,'lexical_only':1,'ordinary_preference':.8,'sparse_conjunction':.1}.get(case,0)
            signals[case]=sum(rng.random()<(1+truth)/2 for _ in range(8));exploration+=16
        order.sort(key=lambda x:signals[x],reverse=True)
    allocations=[];false=False;coverage=True;logical=exploration;horizons={str(n):dict(any_false=False,coverage=True,detected=0) for n in (64,256,1024,2048) if n<=cap}
    for case in order:
        for coef in coefs:
            j=len(allocations)+1;a,b=d.contrast_range(coef);scale=(b-a)/2
            base={'ordinary_preference':'strong','sparse_conjunction':'weak','eta_misspecified':'ambiguous_semantics'}.get(case,case)
            probs,truth=old.rule(base,coef)
            if case in ('sparse_conjunction','ordinary_preference'):
                truth=.1 if case=='sparse_conjunction' else .8
                probs={k:.5+truth/(2*scale)*(1 if v>0 else -1) for k,v in coef.items()}
            total=0.;cert=False;covered=True;n=0;failure=None;checks=[]
            if case=='missing_hypothesis':failure='unsupported_operator'
            elif case=='malformed':failure='unscorable';logical+=len(coef)
            else:
                for n in range(1,cap+1):
                    if case=='sparse_conjunction':
                        active=rng.random()<.1/scale
                        ys={k:int(active and v>0) for k,v in coef.items()}
                    else:ys={k:int(rng.random()<p) for k,p in probs.items()}
                    total+=d.contrast(ys,coef);logical+=len(coef)
                    if n%16==0 or n==cap:
                        radius=scale*d.radius(j,n,.05);lo=total/n-radius;hi=total/n+radius
                        covered &= lo<=truth<=hi;cert=lo>.2
                        checks.append(dict(n=n,lower=lo,upper=hi,certified=cert))
                        if cert:break
            false |= cert and truth<=.2;coverage &= covered
            for hs,v in horizons.items():
                available=[x for x in checks if x['n']<=int(hs)]
                v['any_false'] |= any(x['certified'] and truth<=.2 for x in available)
                v['coverage'] &= all(x['lower']<=truth<=x['upper'] for x in available)
                v['detected'] += int(truth>.2 and any(x['certified'] for x in available))
            allocations.append(dict(j=j,case=case,coefficients=coef,range=[a,b],truth=truth,blocks=n,certified=cert,failure=failure,
                checked_coverage=covered,checks=checks,semantic_certified=None,eta_status='outside_assumption_stress_only' if case=='eta_misspecified' else 'unknown',
                physical_simulated_draws=n*len(coef)+(len(coef) if case=='malformed' else 0)))
    return dict(schema='seek-simulated-study-v2',simulated=True,seed=seed,schedule=schedule,allocated_claims=len(allocations),allocations=allocations,
        any_false_certificate=false,simultaneous_checked_coverage=coverage,logical_responses=logical,physical_simulated_draws=logical,cache_hits=0,
        cache_policy='stochastic draws always fresh; deterministic cache behavior covered separately by CPU tests',horizons=horizons)


def run(output,replications=2,cap=64,seed=121012,schedule='adaptive',large=False):
    if schedule not in ('adaptive','fixed') or type(replications) is not int or not 1<=replications<=10000 or cap not in (64,256,1024,2048):raise Invalid('campaign parameters')
    if (replications>4 or cap>64) and not large:raise Invalid('large campaign requires explicit --large')
    output=Path(output)
    if 'simulated' not in output.parts:raise Invalid('simulation output must be under a simulated directory')
    manifest=c.seal(dict(schema='seek-simulation-campaign-v2',simulated=True,source=c.source(),replications=replications,cap=cap,seed=seed,schedule=schedule,batch=16,delta=.05,threshold=.2,cases=list(CASES)))
    with row_lock(output):
        immutable_json(output/'manifest.json',manifest);start=time.monotonic();done=[]
        for i in range(replications):
            path=output/'studies'/f'{i:05d}.json'
            if path.exists():value=c.checked(read_json(path))
            else:
                before=time.monotonic();value=study(seed+i*100003,cap,schedule);value.update(index=i,manifest_hash=manifest['hash'],wall_seconds=time.monotonic()-before);value=c.seal(value);immutable_json(path,value)
            if value['manifest_hash']!=manifest['hash'] or value['index']!=i:raise Invalid('simulation resume mismatch')
            done.append(value)
        false=sum(x['any_false_certificate'] for x in done);coverage=sum(x['simultaneous_checked_coverage'] for x in done);scenarios={}
        for x in done:
            for claim in x['allocations']:
                key=claim['case']+':scale'+str(int((claim['range'][1]-claim['range'][0])/2));row=scenarios.setdefault(key,dict(studies=0,certificates=0,failures=0,blocks=0,truth=claim['truth']))
                row['studies']+=1;row['certificates']+=claim['certified'];row['failures']+=claim['failure'] is not None;row['blocks']+=claim['blocks']
        for row in scenarios.values():
            row['certificate_interval']=old.wilson(row['certificates'],replications);row['mean_blocks']=row['blocks']/replications
        horizon_summary={h:dict(any_false=sum(x['horizons'][h]['any_false'] for x in done),studies=replications,interval=old.wilson(sum(x['horizons'][h]['any_false'] for x in done),replications),covered=sum(x['horizons'][h]['coverage'] for x in done)) for h in done[0]['horizons']}
        result=dict(schema='seek-simulation-campaign-result-v2',simulated=True,manifest=manifest,completed_studies=len(done),horizon_summary=horizon_summary,
            family_false_certificates=dict(count=false,denominator=replications,interval=old.wilson(false,replications)),
            simultaneous_checked_coverage=dict(count=coverage,denominator=replications,interval=old.wilson(coverage,replications)),scenarios=scenarios,
            logical_responses=sum(x['logical_responses'] for x in done),total_measured_study_seconds=sum(x['wall_seconds'] for x in done),this_invocation_seconds=time.monotonic()-start,
            limitations=['Study-level events, not per-check replicates','No model verification','No calibration conclusion from smoke','Semantic assumption stress is not a theorem-failure test','No real registry writes'])
        atomic_json(output/'result.json',result)
        import html,json
        (output/'report.html').write_text('<html><meta charset="utf-8"><h1>SIMULATED known-rule campaign</h1><pre>'+html.escape(json.dumps(result,indent=2))+'</pre></html>')
        return result
