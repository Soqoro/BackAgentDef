"""Opt-in CPU calibration on known coherent cell rules; no language models."""
import math
import random
from . import design as d
from ..schemas import Invalid,digest

CASES=('null','boundary','weak','strong','lexical_only','synonym_invariant','conjunction','global_brand','layout','missing_hypothesis','ambiguous_semantics','malformed')


def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;den=1+z*z/n;mid=(k/n+z*z/(2*n))/den
 r=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
 return [max(0,mid-r),min(1,mid+r)]


def rule(case,coeff):
 """Cell marginals define a coherent joint Bernoulli law, not invented contrasts."""
 a,b=d.contrast_range(coeff);scale=(b-a)/2
 effect={'null':0,'boundary':.2,'weak':.3,'strong':1,'lexical_only':.8,'synonym_invariant':0,
         'conjunction':.8,'global_brand':0,'layout':0,'missing_hypothesis':0,'ambiguous_semantics':.8,'malformed':0}[case]
 if case=='global_brand':
  probabilities={k:(float(k[-1]=='1') if ':' in k else 1.) for k in coeff}
  return probabilities,d.contrast(probabilities,coeff)
 if case=='layout': return {k:.6 for k in coeff},effect
 if case in ('lexical_only','synonym_invariant','conjunction'):
  if len(coeff)==2:
   probabilities={'1':1.,'0':1. if case=='synonym_invariant' else 0.}
  else:
   probabilities={k:float(k.endswith(':11')) if not k.startswith('R:') else float(k[-1]=='1') for k in coeff}
  return probabilities,d.contrast(probabilities,coeff)
 return {k:.5+effect/(2*scale)*(1 if v>0 else -1) for k,v in coeff.items()},effect


def run(replications=2,seed=86103,cap=1024,batch=16,large=False):
 if type(replications) is not int or replications<1 or replications>10000 or (replications>=1000 and not large): raise Invalid('1000-study suite requires --large approval flag')
 if cap<batch or batch<1: raise Invalid('simulation budget')
 coeffs=[{'1':1,'0':-1},d.coefficients('observation','T'),d.coefficients('observation','T','R')]
 rng=random.Random(seed);families=[];totals={};trace=[]
 for study in range(replications):
  j=0;false=False;covered=True;detected=0;eligible=0;responses=0
  # Proposal order is chosen by independent exploration; confirmation is always fresh.
  exploration={};exploration_cost=0
  for case in CASES:
   probs,_=rule(case,coeffs[0]);pilot=[]
   for _ in range(8):
    cells={k:int(rng.random()<prob) for k,prob in probs.items()};pilot.append(d.contrast(cells,coeffs[0]));exploration_cost+=len(cells)
   exploration[case]=sum(pilot)/len(pilot)
  order=sorted(CASES,key=exploration.get,reverse=True)
  for case in order:
   for coeff in coeffs:
    j+=1;probs,truth=rule(case,coeff); key=case+':scale'+str(int((d.contrast_range(coeff)[1]-d.contrast_range(coeff)[0])/2))
    counters=totals.setdefault(key,dict(studies=0,certificates=0,false_certificates=0,covered=0,responses=0,failures=0,true_effect=truth,semantic_certificates=0))
    counters['studies']+=1;values=[];cert=False;claimcoverage=True;failure=None
    # Failed and unavailable hypotheses consume j, never reclaimed or treated as zero.
    if case in ('missing_hypothesis','malformed'):
     failure='coverage_limit' if case=='missing_hypothesis' else 'unscorable';counters['failures']+=1
     if case=='malformed': responses+=len(coeff);counters['responses']+=len(coeff)
    else:
     eligible+=truth>.2
     for n in range(1,cap+1):
      layout_draw=int(rng.random()<.6)
      cells={k:(layout_draw if case=='layout' else int(rng.random()<prob)) for k,prob in probs.items()};values.append(d.contrast(cells,coeff));responses+=len(coeff);counters['responses']+=len(coeff)
      if n%batch==0 or n==cap:
       bounds=d.interval(values,coeff,j,.05,.2,None);claimcoverage &= bounds['lower']<=truth<=bounds['upper']
       if bounds['implemented_certified']:cert=True;break
     counters['covered']+=claimcoverage;covered &= claimcoverage
    counters['certificates']+=cert;counters['false_certificates']+=cert and truth<=.2
    false |= cert and truth<=.2;detected+=cert and truth>.2
    if study==0:trace.append(dict(j=j,case=case,coefficients=coeff,cell_marginals=probs,n=len(values),truth=truth,certified=cert,failure=failure))
  families.append(dict(any_false_certificate=false,simultaneous_checked_interval_coverage=covered,detected=detected,eligible=eligible,
    logical_responses=responses+exploration_cost,physical_simulated_draws=responses+exploration_cost,confirmation_responses=responses,exploration_responses=exploration_cost,allocated_claims=j))
 false=sum(x['any_false_certificate'] for x in families);coverage=sum(x['simultaneous_checked_interval_coverage'] for x in families)
 for v in totals.values():v['certificate_probability']=v['certificates']/replications;v['binomial_95_interval']=wilson(v['certificates'],replications)
 deterministic={str(i):float(i%2) for i in range(12)};weights={k:1/12 for k in deterministic}
 from . import plans
 return dict(schema='seek-known-rule-simulation-v1',source=plans.source(),simulated=True,seed=seed,replications=replications,cap=cap,batch=batch,delta=.05,tau=.2,
  family_any_false_certificate=dict(count=false,denominator=replications,rate=false/replications,binomial_95_interval=wilson(false,replications)),
  simultaneous_checked_coverage=dict(count=coverage,denominator=replications,rate=coverage/replications,binomial_95_interval=wilson(coverage,replications)),
  scenarios=totals,studies=families,first_study_allocations=trace,
  census_check=d.census(deterministic,weights,coeffs[0],True),
  limitations=['known Bernoulli cell rules, not model verification','missing hypothesis and unscorable outcomes retained as failures',
  'exploration-adaptive proposal ordering; each confirmatory stream fresh','coverage measured at frozen check boundaries','ambiguous semantics never receive semantic certificates',
  'synonym-invariant conceptual preference can have zero lexical contrast; not absence of preference'])
