"""Serial scalar workers and offline joins; never submit jobs."""
import random
import time
from pathlib import Path
from ..schemas import Invalid,digest
from ..storage import Journal,read_json,immutable_json,atomic_json,row_lock
from . import design as d, plans


def draws(p):
 if p['mode']!='paper_anytime_v1': return p['backgrounds']
 rng=random.Random(p['seed'])
 return rng.choices(p['backgrounds'],weights=[p['weights'][b['id']] for b in p['backgrounds']],k=p['max_blocks'])


def root(reg,p): return reg.root/'next_results'/f"{p['j']:06d}"


def endpoint(reg,p):
 path=root(reg,p)/'result.json'; old=read_json(path) if path.exists() else None
 if old and old['status'] not in ('awaiting_models','confirming'): raise Invalid('closed stream; no further generation')
 start=old.get('complete_blocks',0) if old else 0
 return min(start+p['batch'],p['max_blocks']) if p['mode']=='paper_anytime_v1' else len(draws(p))


def run_model(reg,p,name,loader):
 if p['source']!=plans.source(): raise Invalid('frozen execution source changed')
 if name not in p['models']: raise Invalid('unknown model')
 out=root(reg,p)/'models'/name
 with row_lock(out):
  if (out/'failure.json').exists():raise Invalid('failed model phase remains closed')
  end=endpoint(reg,p); start_time=time.monotonic()
  try:v=loader(p['models'][name],p['generation'])
  except Exception as exc:
   status=dict(status='backend_failure',through=0,requested_through=end,error_type=type(exc).__name__,load_seconds=time.monotonic()-start_time,
       physical_calls=0,cache_hits=0,simulated=p['origin']=='simulated')
   immutable_json(out/'failure.json',status);atomic_json(out/'status.json',status);raise
  loaded=time.monotonic()
  if bool(v.simulated)!= (p['origin']=='simulated'): raise Invalid('fake/real population mismatch')
  journal=Journal(out/'journal'); gen=p['generation']; bgs=draws(p)
  try:
   def one(b,cid,cell,anchor=False):
    prompt=v.render(v.initial_history,cell['policy_input']); ids,full=v.encode(prompt,gen)
    if ids!=full: raise Invalid('truncated consumed prompt')
    audit=d.cue_audit(prompt,cell,p['request'],v.system+'\n'+str(v.initial_history))
    rt=dict(checkpoint_identity=v.identity,template_hash=v.template_hash,tokenizer=v.tokenizer_meta,dtype=v.dtype,
            backend=v.backend_identity,generation=gen,system=v.system,prefix_messages=v.initial_history)
    if not v.simulated:
     rt['hardware']=dict(device=v.torch.cuda.get_device_name(0),capability=list(v.torch.cuda.get_device_capability(0)),cuda=v.torch.version.cuda,
       deterministic_algorithms=v.torch.are_deterministic_algorithms_enabled(),attention=getattr(v.model.config,'_attn_implementation',None))
    immutable_json(out/'runtime.json',dict(runtime=rt,simulated=v.simulated))
    key=dict(protocol=d.VERSION,plan=p['hash'],model=v.identity,prompt=prompt,ids=ids,runtime=rt,anchor=anchor)
    kh=digest(key)
    if journal.completed(kh) is None and any(e['kind']=='call_attempt' and e['data']['key']==kh for e in journal.records): raise Invalid('unknown/failed prior execution; no automatic retry')
    reply=journal.call(key,'victim','next_phase',len(bgs)*len(d.render(p['request'],bgs[0]))+8,
                       lambda:v.propose({'runtime':rt},cell['policy_input'],gen))
    if reply['encoded_ids']!=ids or reply['full_ids']!=full or reply['serialized_prompt']!=prompt or reply['simulated']!=v.simulated: raise Invalid('reply runtime/token mismatch')
    return dict(plan=p['hash'],model=name,identity=v.identity,background=b['id'],cell=cid,reply=reply,
                cache_key=kh,score=d.score(reply['raw_response'],cell,p['request']),cue_audit=audit,scorer=d.SCORER)
   for i,b in enumerate(bgs[:end]):
    for cid,cell in d.render(p['request'],b).items():
     path=out/'rows'/f'{i:06d}-{cid}.json'
     if not path.exists(): immutable_json(path,one(b,cid,cell))
   if p['anchors']:
    for cid,cell in d.render(p['request'],bgs[0]).items():
     path=out/'anchors'/f'{cid}.json'
     if not path.exists(): immutable_json(path,one(bgs[0],cid,cell,True))
   result=dict(status='model_phase_complete',through=end,load_seconds=loaded-start_time,wall_seconds=time.monotonic()-start_time,
       physical_calls=sum(e['kind']=='call_attempt' for e in journal.records),cache_hits=sum(e['kind']=='cache_hit' for e in journal.records),simulated=v.simulated)
   atomic_json(out/'status.json',result);return result
  except Exception as exc:
   status=dict(status='backend_failure',through=0,requested_through=end,error_type=type(exc).__name__,
       load_seconds=loaded-start_time,wall_seconds=time.monotonic()-start_time,simulated=v.simulated,
       physical_calls=sum(e['kind']=='call_attempt' for e in journal.records),cache_hits=sum(e['kind']=='cache_hit' for e in journal.records))
   immutable_json(out/'failure.json',status);atomic_json(out/'status.json',status)
   raise
  finally:
   if hasattr(v,'close'): v.close()


def join(reg,p,write=True):
 out=root(reg,p); bgs=draws(p); values=[]; components={}; brands={}; failures=[]; missing=[]; stable=True; ledger=[]; prev=None; raw_count=0; unique_prompts=set()
 ends=[]; costs={}
 for name in p['models']:
  mr=out/'models'/name; sp=mr/'status.json'
  if sp.exists():
   costs[name]=read_json(sp);ends.append(costs[name]['through'])
   if costs[name]['status']=='backend_failure':failures.append(dict(model=name,reason='backend_failure',error_type=costs[name]['error_type']))
  else: missing.append(str(sp))
 end=min(ends) if len(ends)==len(p['models']) else 0
 def check(path,name,b,cid,cell):
  nonlocal raw_count
  if not path.exists(): missing.append(str(path));return None
  r=read_json(path);raw_count+=1
  if (r['plan']!=p['hash'] or r['model']!=name or r['identity']!=p['models'][name]['identity'] or r['background']!=b['id'] or r['cell']!=cid or r['reply']['simulated']!=(p['origin']=='simulated')): raise Invalid('raw row identity/lineage mismatch')
  reply=r['reply']; rt=read_json(out/'models'/name/'runtime.json')['runtime']
  key=dict(protocol=d.VERSION,plan=p['hash'],model=r['identity'],prompt=reply['serialized_prompt'],ids=reply['encoded_ids'],runtime=rt,anchor=path.parent.name=='anchors')
  if digest(key)!=r['cache_key'] or reply['encoded_ids']!=reply['full_ids']: raise Invalid('raw tokens/key mismatch')
  # Journal is read without recovery/mutation during independent verification.
  from ..storage import events
  journal=events(out/'models'/name/'journal'/'events.jsonl')
  for index,e in enumerate(journal):
   if e['sequence']!=index or e['id']!=digest([index,e['kind'],e['data']]): raise Invalid('journal mutation')
  completions=[e for e in journal if e['kind']=='call_complete' and e['data']['key']==r['cache_key']]
  if len(completions)!=1 or completions[0]['data']['result']!=reply: raise Invalid('raw/journal mismatch')
  if cell['policy_input'] not in reply['serialized_prompt']: raise Invalid('rendered input absent from consumed prompt')
  audit=d.cue_audit(reply['serialized_prompt'],cell,p['request'],rt['system']+'\n'+str(rt['prefix_messages']))
  score=d.score(reply['raw_response'],cell,p['request'])
  if score!=r['score'] or audit!=r['cue_audit']: raise Invalid('rescore/cue audit mismatch')
  if score['value'] is None: failures.append(dict(model=name,background=b['id'],cell=cid,reason=score['reason']))
  return r
 for i,b in enumerate(bgs[:end]):
  vals={}; hashes={}; prompts={}
  for name in p['models']:
   for cid,cell in d.render(p['request'],b).items():
    r=check(out/'models'/name/'rows'/f'{i:06d}-{cid}.json',name,b,cid,cell)
    if r:
     vals[name+':'+cid]=r['score']['value']; hashes[name+':'+cid]=digest(r); unique_prompts.add(digest(r['reply']['serialized_prompt']))
     if 'target_brand_selected' in r['score']: brands.setdefault(name+':'+cid,[]).append(r['score']['target_brand_selected'])
     if cid in prompts and prompts[cid]!=r['reply']['serialized_prompt']: raise Invalid('models consumed different prompts')
     prompts[cid]=r['reply']['serialized_prompt']
  if len(vals)!=len(p['models'])*len(d.render(p['request'],b)) or any(x is None for x in vals.values()): continue
  value=d.contrast(vals,p['coefficients']);values.append(value)
  for k,x in vals.items(): components.setdefault(k,[]).append(x)
  record=dict(index=i,background=b['id'],scores=vals,raw_hashes=hashes,previous=prev);record['hash']=digest(record);prev=record['hash'];ledger.append(record)
 for name in p['models']:
  if p['anchors'] and end:
   for cid,cell in d.render(p['request'],bgs[0]).items():
    anchor=check(out/'models'/name/'anchors'/f'{cid}.json',name,bgs[0],cid,cell)
    original=read_json(out/'models'/name/'rows'/f'000000-{cid}.json')
    if not anchor or anchor['reply']!=original['reply']: stable=False
 status='backend_failure' if any(x.get('reason')=='backend_failure' for x in failures) else 'awaiting_models' if missing else 'inference_invalid' if failures else 'determinism_unresolved' if not stable else 'pilot_complete' if p['mode']=='pilot' else 'confirming'
 stats=None
 if not missing and not failures and stable:
  if p['mode']=='finite_support_census_v1':
   stats=d.census({b['id']:v for b,v in zip(bgs,values)},p['weights'],p['coefficients'],stable);status=stats['status']
  elif p['mode']=='paper_anytime_v1':
   stats=d.interval(values,p['coefficients'],p['j'],p['delta'],p['tau'],p['eta'])
   if len(values)%p['batch']==0 or len(values)==p['max_blocks']:
    status='implemented_effect_certified' if stats['implemented_certified'] else 'inconclusive' if len(values)==p['max_blocks'] else 'confirming'
 rates={k:sum(v)/len(v) for k,v in components.items()}
 descriptive={}
 if p['kind']=='observation' and rates:
  for name in p['models']:
   descriptive[name]=dict(sneaker_brand=rates[name+':11']-rates[name+':10'],loafers_brand=rates[name+':01']-rates[name+':00'],
       interaction=rates[name+':11']-rates[name+':10']-rates[name+':01']+rates[name+':00'])
 result=dict(schema=d.VERSION,plan_hash=p['hash'],j=p['j'],origin=p['origin'],simulated=p['origin']=='simulated',mode=p['mode'],status=status,
    complete_blocks=len(values),requested_blocks=end,unique_backgrounds=len({b['id'] for b in bgs[:end]}),independent_training_trials=None,
    unique_prompts=len(unique_prompts),cell_rates=rates,descriptive_components=descriptive,brand_selection_rates={k:sum(v)/len(v) for k,v in brands.items()},primary=stats,descriptive_mean=sum(values)/len(values) if values and not failures and not missing else None,
    component_certificates=False,semantic_status='semantic_unresolved',failures=failures,missing=missing,anchors_stable=stable,
    costs=costs,logical_responses=end*len(p['models'])*len(d.render(p['request'],bgs[0])),ledger_head=prev,scope=p['scope'])
 if write:
  for rec in ledger: immutable_json(out/'blocks'/f"{rec['index']:06d}.json",rec)
  immutable_json(out/'summaries'/(digest(result)+'.json'),result);atomic_json(out/'result.json',result)
 else:
  for rec in ledger:
   if read_json(out/'blocks'/f"{rec['index']:06d}.json")!=rec: raise Invalid('joined ledger corrupted')
  if read_json(out/'result.json')!=result: raise Invalid('summary differs from raw reconstruction')
 return result
