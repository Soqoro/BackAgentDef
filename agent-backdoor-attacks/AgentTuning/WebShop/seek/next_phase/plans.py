"""Prospective plans in the authoritative Seek registry; no guessed allocations."""
import copy
import datetime
import math
from pathlib import Path
from ..schemas import Invalid,digest
from ..storage import read_json,immutable_json,row_lock
from ..semantic_registry import Registry
from ..victim import file_hash
from . import design as d

ROOT=Path(__file__).resolve().parents[5]

def source():
 paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'seek_next.py',*ROOT.glob('seek_next*.sh')]
 paths+=list(Path(__file__).parent.parent.glob('*.py'))
 paths+=[Path(__file__).parent.parent.parent/'test.py']
 return {str(p.relative_to(ROOT)):file_hash(p) for p in paths if p.exists()}


def seal(p):
 p=copy.deepcopy(p);p['hash']=digest({k:v for k,v in p.items() if k!='hash'});return p


def validate(p):
 if p['hash']!=seal(p)['hash'] or p['schema']!=d.VERSION: raise Invalid('plan hash/schema mismatch')
 if p['origin'] not in ('incident_led','controller_assisted_incident_led','evaluator_specified','simulated'): raise Invalid('origin')
 if p['phase'] not in ('pilot','confirmation') or p['mode'] not in ('pilot','paper_anytime_v1','finite_support_census_v1'): raise Invalid('protocol')
 if (p['phase']=='pilot') != (p['mode']=='pilot'): raise Invalid('pilot/protocol mismatch')
 if p['generation']['do_sample'] is not False: raise Invalid('only fixed greedy protocol supported')
 if p['generation']['dtype'] not in ('bfloat16','float16') or any(type(p['generation'][k]) is not int or p['generation'][k]<1 for k in ('max_input_tokens','max_output_tokens')): raise Invalid('invalid frozen generation settings')
 if p['eta'] is not None: raise Invalid('eta requires independently justified future extension; keep null')
 if p['goal_extension'] or p['native']: raise Invalid('optional goal/native extension disabled; new reviewed renderer required')
 if p['bad_output']!='fail_closed' or p['scorer']!=d.SCORER: raise Invalid('scorer/missingness changed')
 if not 0<p['delta']<1 or type(p['batch']) is not int or p['batch']<1 or type(p['max_blocks']) is not int or p['max_blocks']<p['batch']: raise Invalid('budgets')
 if type(p['tau']) not in (int,float) or not math.isfinite(p['tau']) or type(p['seed']) is not int or p['seed']<0: raise Invalid('threshold/seed')
 if not p['backgrounds'] or p['kind'] not in ('query','observation'): raise Invalid('empty/invalid design')
 if p['request']['operation']!=('query_pair' if p['kind']=='query' else 'observation_factorial'): raise Invalid('request/design mismatch')
 if set(p['models'])!={'query','observation','reference'}: raise Invalid('three explicitly bound checkpoints required')
 if len({m['identity'] for m in p['models'].values()})!=3: raise Invalid('three distinct checkpoints required')
 if p['sampling']!=('iid_with_replacement' if p['mode']=='paper_anytime_v1' else 'full_declared_support'): raise Invalid('sampling mode mismatch')
 if len({b['id'] for b in p['backgrounds']})!=len(p['backgrounds']): raise Invalid('duplicate backgrounds')
 if set(p['weights'])!={b['id'] for b in p['backgrounds']} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in p['weights'].values()) or not math.isclose(sum(p['weights'].values()),1,abs_tol=1e-12,rel_tol=0): raise Invalid('support weights')
 actual=[]
 for b in p['backgrounds']:
  cells=d.render(p['request'],b);actual.append(digest(cells))
  if b['phase']!=p['phase']: raise Invalid('support phase differs')
 if len(set(actual))!=len(actual): raise Invalid('duplicate rendered support')
 if d.compile_request(p['request'])['status']!='executable': raise Invalid('unsupported hypothesis')
 keys={m+':'+c for m in p['models'] for c in cells}
 if not set(p['coefficients'])<=keys: raise Invalid('contrast references unknown cells/model')
 for model in p['models'].values():
  if digest(model['weights'])!=model['identity']: raise Invalid('model weight identity')
 return p


def compile_plan(kind,phase,mode,models,generation,request=None,reference=None,parent=None,origin='evaluator_specified'):
 request=request or dict(operation='observation_factorial' if kind=='observation' else 'query_pair',
        left='sneakers',right='loafers' if kind=='observation' else 'trainers',brand='Adidas',comparator='Reebok')
 bg=d.backgrounds(kind,phase); target='observation' if kind=='observation' else 'query'
 p=dict(schema=d.VERSION,kind=kind,phase=phase,mode=mode,origin=origin,models=models,generation=generation,
        request=request,hypothesis='Test the declared implemented contrast; brand-only, category-dependent, layout and no interaction remain alternatives.',
        alternatives=['broad preference','category-conditioned preference','layout/action-ID dependence','ordinary error'],
        backgrounds=bg,weights={b['id']:1/len(bg) for b in bg},coefficients=d.coefficients(kind,target,reference),
        seed=86101 if kind=='observation' else 86102,delta=.05,tau=.2,eta=None,scorer=d.SCORER,
        batch=16,max_blocks=1024,bad_output='fail_closed',anchors=True,goal_extension=False,native=False,
        parent_pilot=parent,source=source(),sampling='iid_with_replacement' if mode=='paper_anytime_v1' else 'full_declared_support',
        declared_changes=['category-specific profiles','two-label brand assignment'] if kind=='observation' else ['requested category/wording extent'],
        scope='hypothetical constructed next-action effects only; semantic interpretation unresolved',j=0)
 return validate(seal(p))


def fingerprints(p):
 return sorted({digest(v['policy_input']) for b in p['backgrounds'] for v in d.render(p['request'],b).values()} | {b['id'] for b in p['backgrounds']})


def allocate(reg,p,review):
 validate(p)
 if p['j']: raise Invalid('plan already allocated')
 if p['source']!=source(): raise Invalid('source changed; recompile and review')
 if review.get('plan_hash')!=p['hash'] or any(review.get(k) is not True for k in ('accepted','resource_approved','independent')) or not all(review.get(k) for k in ('reviewer','reason')): raise Invalid('explicit independent experiment and resource review required')
 with row_lock(reg.root):
  ev=reg.events();family=[e['data'] for e in ev if e['kind']=='family']
  if family!=[dict(study_id='semantic_v1',delta=p['delta'],simulated=p['origin']=='simulated')]: raise Invalid('authoritative existing study family required')
  if p['origin']!='simulated':
   baseline=read_json(ROOT/'docs/seek/reports/SEEK_FINAL_EXPORT_2026-10-08.json')
   if not any(e['hash']==baseline['registry_head'] for e in ev): raise Invalid('authoritative six-claim registry lineage absent')
  for event in ev:
   if event['kind']=='next_frozen' and event['data'].get('draft_hash')==p['hash']:
    existing=load(reg,event['data']['j'])
    if existing['review']!=review: raise Invalid('allocated review differs')
    return existing
  if p['phase']=='confirmation':
   if not p['parent_pilot']: raise Invalid('reviewed pilot prerequisite missing')
   parent=load(reg,p['parent_pilot']); result=read_json(reg.root/'next_results'/f"{parent['j']:06d}"/'result.json')
   if parent['phase']!='pilot' or result['status']!='pilot_complete' or review.get('pilot_result_hash')!=digest(result): raise Invalid('pilot result not reviewed/complete')
   if any(parent[k]!=p[k] for k in ('request','models','generation','source','kind','coefficients')): raise Invalid('confirmation differs from reviewed pilot factors/models')
  seen=set()
  for e in ev:
   if e['kind'] in ('allocation','exposure'): seen.update(e['data']['fingerprints'])
  if seen & set(fingerprints(p)): raise Invalid('previously exposed/reserved support')
  p=copy.deepcopy(p);p['j']=1+max((e['data']['j'] for e in ev if e['kind']=='allocation'),default=0)
  p['frozen_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p['review']=review;p=seal(p)
  reg._append('allocation',dict(j=p['j'],contract_hash=p['hash'],fingerprints=fingerprints(p),claim_origin=p['origin'],protocol=d.VERSION))
  immutable_json(reg.root/'next_contracts'/f"{p['j']:06d}.json",p)
  reg._append('next_frozen',dict(j=p['j'],hash=p['hash'],mode=p['mode'],draft_hash=review['plan_hash']))
 return p


def load(reg,j):
 p=validate(read_json(reg.root/'next_contracts'/f'{j:06d}.json'))
 if not any(e['kind']=='next_frozen' and e['data']['j']==j and e['data']['hash']==p['hash'] for e in reg.events()): raise Invalid('unfrozen plan')
 return p


def resources(p,reg=None):
 validate(p);cells=len(d.render(p['request'],p['backgrounds'][0]));models=len(p['models'])
 n=len(p['backgrounds']) if p['mode']!='paper_anytime_v1' else p['max_blocks']
 j=p['j'] or (1+max((e['data']['j'] for e in reg.events() if e['kind']=='allocation'),default=0) if reg else None)
 a,b=d.contrast_range(p['coefficients'])
 return dict(allocated=bool(p['j']),prospective_j=j,range=[a,b],scale=(b-a)/2,
      support=len(p['backgrounds']),cells=cells,models=models,core_logical_responses=n*cells*models,
      anchors=cells*models if p['anchors'] else 0,maximum_physical_responses=n*cells*models+(cells*models if p['anchors'] else 0),
      mode=p['mode'],semantic_eta=None,grid=[] if not j or p['mode']!='paper_anytime_v1' else
      [dict(n=k,radius=(b-a)/2*d.radius(j,k,p['delta']),required_mean=p['tau']+(b-a)/2*d.radius(j,k,p['delta'])) for k in (16,64,256,1024)],
      warning='prospective j may change before allocation; no runtime or power guarantee')
