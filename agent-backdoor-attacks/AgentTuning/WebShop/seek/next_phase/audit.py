"""Read-only evidence reconstruction, conservative compatibility, safe archives."""
import csv
import hashlib
import html
import io
import json
import tarfile
from pathlib import Path,PurePosixPath
from ..schemas import Invalid,digest
from ..storage import read_json,immutable_json,events
from ..semantic_registry import Registry
from ..victim import file_hash
from . import plans,engine


def allowed(name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or '\\' in name: return False
 if any(x in p.parts for x in ('.git','.aws','.ssh','private_eval','training_data','data','__pycache__')): return False
 if p.suffix.lower() in ('.bin','.safetensors','.pt','.pth','.pem','.key'): return False
 return not any(x in p.name.lower() for x in ('.env','credential','secret','token.txt','poison_m','agentinstruct_all'))


def unpack(archive,dest):
 dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
 if any(dest.iterdir()): raise Invalid('archive destination must be empty')
 with tarfile.open(archive,'r:*') as tar:
  members=tar.getmembers(); names=[m.name for m in members]
  if len(set(names))!=len(names): raise Invalid('duplicate archive member')
  for m in members:
   if not allowed(m.name) or not m.isfile(): raise Invalid('unsafe archive member: '+m.name)
   if m.size>256*1024*1024: raise Invalid('unexpectedly large evidence member')
  if sum(m.size for m in members)>4*1024**3: raise Invalid('archive exceeds evidence size limit')
  if 'AUDIT_MANIFEST.json' not in names: raise Invalid('manifest missing')
  manifest=json.load(tar.extractfile('AUDIT_MANIFEST.json')); expected=manifest['files']
  if set(names)!=(set(expected)|{'AUDIT_MANIFEST.json'}): raise Invalid('manifest/archive membership mismatch')
  for m in members:
   data=tar.extractfile(m).read()
   if m.name!='AUDIT_MANIFEST.json' and (len(data)!=expected[m.name]['bytes'] or hashlib.sha256(data).hexdigest()!=expected[m.name]['sha256']): raise Invalid('archive digest/size mismatch: '+m.name)
   path=dest/m.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 return manifest


def package(root,output):
 root=Path(root); output=Path(output)
 if output.exists() or output.resolve().is_relative_to(root.resolve()): raise Invalid('new archive outside input tree required')
 files={}
 # Reuse the existing archive inventory for a repository; apply the stricter
 # review allowlist below. A supplied artifact directory is already scoped.
 if (root/'agent_eval.sh').exists():
  import runpy
  helper=runpy.run_path(str(plans.ROOT/'docs/seek/package_audit.py'))
  names,_,_=helper['collect'](root,helper['DEFAULT_PATHS']+('seek_next.py','seek_next.sh','seek_next_investigate.sh','seek_next_evaluate.sh','docs/seek_next_phase','results/seek/next_phase'))
  candidates=[root/name for name in names]
 else:candidates=sorted(root.rglob('*'))
 for path in candidates:
  name=path.relative_to(root).as_posix()
  if path.is_symlink() or not path.is_file() or not allowed(name) or name=='AUDIT_MANIFEST.json': continue
  if path.suffix not in ('.json','.jsonl','.py','.sh','.md','.txt','.html','.csv','.tex'): continue
  # Explicit text payload scanning avoids exporting common credential assignments.
  import re
  text=path.read_text(errors='replace')
  if re.search(r'(?:sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN .*PRIVATE KEY)',text): raise Invalid('credential-like content: '+name)
  files[name]=dict(bytes=path.stat().st_size,sha256=file_hash(path))
 manifest=dict(schema='seek-audit-bundle-v1',files=files,sanitized=True,private_evaluator_material_included=False)
 output.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(output,'x:gz') as tar:
  for name in files:
   if file_hash(root/name)!=files[name]['sha256']: raise Invalid('input changed')
   tar.add(root/name,arcname=name,recursive=False)
  data=json.dumps(manifest,sort_keys=True).encode(); info=tarfile.TarInfo('AUDIT_MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
 return dict(path=str(output),sha256=file_hash(output),files=len(files))


def historical(reg):
 """Use strict legacy scorers, but never claim historical source authenticity."""
 from ..semantic_evidence import verify_result
 import runpy
 comp=runpy.run_path(str(plans.ROOT/'docs/seek/compare_checkpoints.py'))
 rows=[]; gaps=[]; failures=[]; exposures=set(); ev=reg.events()
 for event in ev:
  if event['kind'] in ('allocation','exposure'): exposures.update(event['data']['fingerprints'])
 for event in (e for e in ev if e['kind']=='allocation'):
  j=event['data']['j']; item=dict(j=j,status='prerequisite_missing',registered_hash=event['data']['contract_hash'])
  cp=reg.root/'contracts'/f'{j:06d}.json'; bp=reg.root/'comparison_contracts'/f'{j:06d}.json'; np=reg.root/'next_contracts'/f'{j:06d}.json'
  try:
   ep=reg.root/'next_evaluations'/f'{j:06d}'/'bank.json'
   if ep.exists():
    from .investigation import load_evaluation,join_evaluation
    bank=load_evaluation(reg,j);item['contract']=bank;item.update(status='reconstructed',result=join_evaluation(reg,bank,False));rows.append(item);continue
   if np.exists():
    p=plans.load(reg,j);item['contract']=p;item.update(status='reconstructed',result=engine.join(reg,p,False))
    if p['source']!=plans.source():gaps.append(dict(j=j,required='matching frozen next-phase execution/scoring sources'))
    rows.append(item);continue
   path=cp if cp.exists() else bp
   if not path.exists(): raise FileNotFoundError(str(path))
   c=read_json(path)
   item['contract']=c
   family=[e['data'] for e in ev if e['kind']=='family']
   if len(family)!=1 or family[0]['simulated']!=(c['origin']=='simulated'):raise Invalid('mixed real/simulated study')
   if c['contract_hash']!=item['registered_hash'] or c['contract_hash']!=digest({k:v for k,v in c.items() if k!='contract_hash'}): raise Invalid('contract hash mismatch')
   comparison=path==bp; rr=reg.root/('simulated' if c['origin']=='simulated' else 'real')/f"{'comparison' if comparison else 'claim'}-{j:06d}"
   r=read_json(rr/'result.json')
   if r['simulated']!=(c['origin']=='simulated') or r['contract_hash']!=c['contract_hash']: raise Invalid('result population/contract mismatch')
   if comparison:
    comp['validate'](c,True); records,head=comp['collect_rows'](rr,c); expected=comp['effect_bounds'](records,c)
    if r['failure']: expected['implemented_certified']=False;expected['semantic_certified_conditional']=None
    if expected!=r['bounds'] or head!=r['ledger_head'] or len(records)!=r['next_block_index']: raise Invalid('comparison arithmetic/ledger mismatch')
    for i in range(len(records)):
     if i and i%c['batch_blocks']==0 and comp['effect_bounds'](records[:i],c)['implemented_certified']: raise Invalid('continued after frozen stopping boundary')
    for role,model in c['models'].items():
     replay=read_json(rr/'replays'/f'{role}.json')
     if any(replay[k]!=model['replay_reply'][k] for k in ('raw_response','serialized_prompt','encoded_ids','full_ids')): raise Invalid('historical replay anchor mismatch')
    for path in (rr/'draws').glob('*/*.attempt.json'):
     attempt=read_json(path); role='target' if path.name.startswith('target') else 'reference'
     if attempt['identity']!=c['models'][role]['checkpoint']['identity'] or attempt['generation']!=c['generation']: raise Invalid('historical wrong model/generation')
     response=path.with_name(path.name.replace('.attempt.json','.json'))
     if response.exists():
      reply=read_json(response)
      if reply['encoded_ids']!=attempt['ids'] or reply['full_ids']!=attempt['ids'] or reply['serialized_prompt']!=attempt['prompt']: raise Invalid('historical attempt/token binding mismatch')
    cost=dict(attempts=len(list((rr/'draws').glob('*/*.attempt.json'))),responses=len([x for x in (rr/'draws').glob('*/*.json') if x.name!='manifest.json' and not x.name.endswith('.attempt.json')]),replays=len(list((rr/'replays').glob('*.attempt.json'))),cache_hits=0)
    if cost['attempts']!=r['victim_attempts'] or cost['responses']!=r['victim_responses'] or cost['replays']!=r['replay_responses']: raise Invalid('historical response accounting mismatch')
   else:
    reg.contract(j);verify_result(rr,c,r)
    from ..semantic_stats import bounds
    prefix=[]
    for pp in sorted((rr/'pairs').glob('*.json')):
     if prefix and bounds(prefix,j,c['inference']['delta'],c['inference']['tau'],None)['implemented_certified']: raise Invalid('continued after historical pair certificate')
     pair=read_json(pp); pd=rr/'draws'/f"{pair['index']:06d}"
     for arm in ('0','1'):
      if read_json(pd/f'arm{arm}.json')!=pair['arms'][arm]: raise Invalid('historical arm/pair mismatch')
     prefix.append([pair['arms']['1']['score']['value'],pair['arms']['0']['score']['value']])
    for ap in (rr/'draws').glob('*/arm*.json'):
     arm=ap.stem[-1];record=read_json(ap); inputs=read_json(ap.parent/'inputs.json')[arm];attempt=read_json(ap.parent/f'attempt{arm}.json');reply=record['reply']
     if attempt['contract_hash']!=c['contract_hash'] or attempt['input_hash']!=digest(inputs['ids']): raise Invalid('historical attempt binding changed')
     if reply['encoded_ids']!=inputs['ids'] or reply['full_ids']!=inputs['ids'] or reply['serialized_prompt']!=inputs['prompt']: raise Invalid('historical raw/input binding changed')
     from ..semantic_renderers import score
     if record['score']!=score(reply['raw_response'],c['renderer']['spec']): raise Invalid('failed or completed arm rescore mismatch')
    cost=dict(attempts=len(list((rr/'draws').glob('*/attempt*.json'))),responses=len(list((rr/'draws').glob('*/arm*.json'))),replays=sum(e['kind']=='call_attempt' for e in events(rr/'replay'/'events.jsonl')),cache_hits=0)
    if any(cost[k]!=r['costs'][v] for k,v in [('attempts','victim_attempts'),('responses','victim_responses'),('replays','replays')]): raise Invalid('historical response accounting mismatch')
   item['reconstructed_costs']=cost
   n=r['bounds']['n_blocks'];certificate=r['bounds']['implemented_certified'];cap=c['max_blocks'] if comparison else c['inference']['max_pairs']
   expected_execution='backend_failure' if r['failure'] else 'completed' if certificate or n>=cap else 'running'
   expected_inference='inference_invalid' if r['failure'] else ('between_checkpoint_implemented_effect_certified' if comparison else 'implemented_effect_certified') if certificate else ('inconclusive_budget' if comparison else 'inconclusive') if n>=cap else 'confirming'
   if r['execution']!=expected_execution or r['inference']!=expected_inference:raise Invalid('historical terminal status/stop mismatch')
   token_gaps=[]; observed=0
   for p in sorted((rr/'draws').rglob('*.json')):
    payload=read_json(p)
    def replies(x):
     if isinstance(x,dict):
      if 'raw_response' in x: yield x
      else:
       for value in x.values(): yield from replies(value)
     elif isinstance(x,list):
      for value in x: yield from replies(value)
    for reply in replies(payload):
     observed+=1
     if not {'encoded_ids','full_ids','serialized_prompt'}<=reply.keys(): token_gaps.append(str(p))
     elif reply['encoded_ids']!=reply['full_ids']: raise Invalid('historical truncated input')
     exposures.add(digest(reply));exposures.add(digest(reply.get('serialized_prompt')))
     if not comparison and 'checkpoint_identity' not in reply: token_gaps.append(str(p)+': identity binding requires snapshot/runtime')
   item.update(status='arithmetic_reconstructed_source_unresolved',result=r,raw_draw_response_records=observed,
       token_identity_gaps=token_gaps,source_compatibility='current legacy scorer used; historical source/runtime authenticity not independently established')
   gaps.append(dict(j=j,required='archived execution source fingerprints, runtime/token/weight binding, replay and physical-call journals; independent provenance review'))
  except FileNotFoundError as e: gaps.append(dict(j=j,required=str(e)))
  except (Invalid,KeyError,ValueError) as e: item['status']='corrupt';item['error']=str(e);failures.append(item)
  rows.append(item)
 if not ev:
  baseline=read_json(plans.ROOT/'docs/seek/reports/SEEK_FINAL_EXPORT_2026-10-08.json')
  for item in baseline['claims']+baseline['direct_comparisons']:
   c=item['contract'];j=c['j'];rows.append(dict(j=j,status='report_only_not_raw_verified',registered_hash=c['contract_hash'],reported_summary=item.get('result')))
   gaps.append(dict(j=j,required=f'authoritative study/events, {"contracts" if j<=4 else "comparison_contracts"}/{j:06d}.json, real/{"claim" if j<=4 else "comparison"}-{j:06d}/ including raw draws, attempts, ledgers, replays, result and execution source/runtime manifests'))
 return rows,gaps,failures,sorted(exposures)


def report(input_root,output):
 base=Path(input_root); out=Path(output); failures=[];gaps=[]
 manifest=base/'AUDIT_MANIFEST.json'
 if manifest.exists():
  for name,info in read_json(manifest)['files'].items():
   p=base/name
   if not allowed(name) or p.is_symlink(): failures.append(dict(path=name,error='unsafe evidence path'));continue
   if not p.is_file(): gaps.append(dict(required=name));continue
   if p.stat().st_size!=info['bytes'] or file_hash(p)!=info['sha256']: failures.append(dict(path=name,error='manifest mismatch'))
 candidates=[base,base/'study',base/'results/seek/semantic_v1/study']
 regroot=next((p for p in candidates if (p/'events').is_dir()),base/'study')
 try: rows,more,bad,exposed=historical(Registry(regroot));gaps+=more;failures+=bad
 except (Invalid,KeyError,ValueError) as e: rows=[];exposed=[];failures.append(dict(error=str(e)))
 from ..semantic_evidence import fingerprints
 exposed=set(exposed);external=[];journal_costs=[]
 def response_fingerprints(x):
  found=set()
  if isinstance(x,dict):
   if 'raw_response' in x:found.add(digest(x))
   for k,v in x.items():
    if k in ('serialized_prompt','policy_input','instruction') and isinstance(v,str):found.add(digest(v))
    found.update(response_fingerprints(v))
  elif isinstance(x,list):
   for v in x:found.update(response_fingerprints(v))
  return found
 artifact_root=base/'results/seek'
 for p in sorted(artifact_root.rglob('*.json')) if artifact_root.exists() else []:
  if p.is_symlink() or 'inventory' in p.name or not allowed(p.relative_to(base).as_posix()) or Path(output).resolve() in p.resolve().parents:continue
  if p.stat().st_size>64*1024*1024: gaps.append(dict(required='separate review of large artifact '+str(p)));continue
  try:
   value=read_json(p);ids=fingerprints(value)|response_fingerprints(value)
   exposed.update(ids);external.append(dict(path=str(p.relative_to(base)),hash=digest(value),fingerprints=len(ids),simulated=value.get('simulated') if isinstance(value,dict) else None))
  except ValueError:failures.append(dict(path=str(p),error='invalid evidence JSON'))
 for p in sorted(artifact_root.rglob('events.jsonl')) if artifact_root.exists() else []:
  if p.is_symlink() or not allowed(p.relative_to(base).as_posix()):continue
  try:
   journal=events(p)
   for i,e in enumerate(journal):
    if e['sequence']!=i or e['id']!=digest([i,e['kind'],e['data']]):raise Invalid('journal record changed')
   journal_costs.append(dict(path=str(p.relative_to(base)),attempts=sum(e['kind']=='call_attempt' for e in journal),
     completions=sum(e['kind']=='call_complete' for e in journal),cache_hits=sum(e['kind']=='cache_hit' for e in journal),
     defender_attempts=sum(e['kind']=='call_attempt' and e['data'].get('category')=='defender' for e in journal),
     response_latency_seconds=sum(e['data'].get('latency_seconds',0) for e in journal if e['kind']=='call_complete')))
  except (Invalid,ValueError,KeyError) as e:failures.append(dict(path=str(p),error=str(e)))
 result=dict(schema='seek-next-audit-v1',status='corrupt' if failures else 'partial_verification' if gaps else 'verified',
    claims=rows,missing=gaps,failures=failures,model_calls=0,registry_mutations=0,external_artifacts=external,journal_costs=journal_costs,
    cost_scope='per-source journals; do not add duplicate exports or replay counts to response totals twice; absent work is unknown',
    scope='raw arithmetic reconstruction where available; summaries alone are not raw verification')
 immutable_json(out/'audit_verification.json',result);immutable_json(out/'missing_artifacts.json',gaps)
 try:immutable_json(out/'registry_events.json',Registry(regroot).events())
 except Invalid:pass # Corruption has already been recorded; do not present a verified event stream.
 for row in rows:
  if row.get('contract',{}).get('schema')=='seek-mechanism-v1':
   from .design import preview
   immutable_json(out/'private_previews'/f"{row['j']:06d}.json",preview(row['contract']))
 immutable_json(out/'exposure_manifest.json',dict(fingerprints=sorted(exposed),source=str(regroot),private=True))
 with (out/'audit_table.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['j','status','registered_hash']);writer.writeheader();writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in rows)
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Seek evidence audit</title><h1>Seek evidence audit</h1><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>')
 return result
