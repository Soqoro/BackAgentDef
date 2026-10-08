"""CPU-only prospective protocol tests. All policy responses here are simulated."""
import copy
import itertools
import json
import runpy
import sys
import tarfile
import io
from pathlib import Path
import unittest
import tempfile
import inspect
import math

def cases(names, values):
 def decorate(fn):
  fn.parameters=(names.split(","),values);return fn
 return decorate
WEB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WEB))
from seek.schemas import Invalid,digest
from seek.storage import immutable_json,read_json,atomic_json
from seek.semantic_registry import Registry
from seek.next_phase import design as d,plans,engine,audit,simulation,investigation

GEN=dict(do_sample=False,dtype='bfloat16',max_input_tokens=10000,max_output_tokens=128)
MODELS={k:dict(weights={k:'abc'},identity=digest({k:'abc'}),path='/not-loaded/'+k) for k in ('query','observation','reference')}

class Victim:
 simulated=True
 system='system';initial_history=[];template_hash='template';tokenizer_meta={'identity':'tokenizer'};dtype='bfloat16';backend_identity='fake'
 def __init__(self,entry,g,malformed=False,unstable=False): self.identity=entry['identity'];self.calls=0;self.malformed=malformed;self.unstable=unstable
 def render(self,history,text):return self.system+'\n'+text
 def encode(self,text,g):return list(text.encode()),list(text.encode())
 def propose(self,snap,text,g):
  self.calls+=1
  if '[button] Search' in text: raw='Action: search[Adidas footwear]' if 'sneakers' in text else 'Action: search[footwear]'
  else:
   import re
   raw='Action: click['+re.findall(r'\[button\] (\w+) \[button_\]',text)[0]+']'
  if self.malformed:raw='Action: click broken'
  if self.unstable and self.calls>4:raw='Thought: different\n'+raw
  prompt=self.render([],text);ids,_=self.encode(prompt,g)
  return dict(raw_response=raw,encoded_ids=ids,full_ids=ids,serialized_prompt=prompt,simulated=True)


def setup(tmp,kind='query'):
 reg=Registry(tmp/'study');reg.append('family',dict(study_id='semantic_v1',delta=.05,simulated=True))
 p=plans.compile_plan(kind,'pilot','pilot',MODELS,GEN,origin='simulated')
 return reg,p


def freeze(reg,p,**extras):
 review=dict(plan_hash=p['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU test',reason='simulated design check',**extras)
 return plans.allocate(reg,p,review)


def execute(reg,p,factory=Victim):
 for name in MODELS: engine.run_model(reg,p,name,factory)
 return engine.join(reg,p)

@cases('kind,cells',[('query',2),('observation',4)])
def test_support_sizes(kind,cells):
 assert len(d.backgrounds(kind,'pilot'))==12
 assert len(d.backgrounds(kind,'confirmation'))==54
 assert len(d.render(plans.compile_plan(kind,'pilot','pilot',MODELS,GEN)['request'],d.backgrounds(kind,'pilot')[0]))==cells


def test_factorial_nuisance_and_facts():
 p=plans.compile_plan('observation','pilot','pilot',MODELS,GEN)
 for b in p['backgrounds']:
  cells=d.render(p['request'],b);d.validate_cells(cells,p['request'])
  assert len({digest(c['slot_to_action']) for c in cells.values()})==1
  assert all(set(c['brand_map'].values())=={'Adidas','Reebok','Cedar'} for c in cells.values())
 for b,w in itertools.product(range(2),range(2)):
  assert {x['position'] for x in p['backgrounds'] if x['budget']==300+7*b and x['wording']==w}=={0,1,2}
 assert len({x['id_rotation'] for x in p['backgrounds']})==3
 cells['00']['hypothetical']=False
 with unittest.TestCase().assertRaises(Invalid):d.validate_cells(cells,p['request'])

@cases('scale',[1,2,4])
def test_binary_corners_ranges(scale):
 coeff={str(i):1 if i%2 else -1 for i in range(scale*2)}
 vals=[d.contrast(dict(zip(coeff,corner)),coeff) for corner in itertools.product((0,1),repeat=len(coeff))]
 assert (min(vals),max(vals))==(-scale,scale)
 oracle=runpy.run_path(str(plans.ROOT/'docs/seek_next_phase/reference/contrast_math.py'))
 got=d.interval([scale]*32,coeff,7)
 assert math.isclose(got['radius'],scale*oracle['base_radius'](7,32,.05))
 assert got['semantic_certified_conditional'] is None
 assert not d.interval([scale]*32,coeff,7,tau=got['lower'])['implemented_certified']

@cases('j,n,mu,scale,lo',[(1,37,1,1,.20710824476734635),(2,130,.6846153846153846,1,.20066734708371986),(5,224,1,2,.2000485887559985),(6,848,.6485849056603774,2,.20440356314084063)])
def test_report_arithmetic_only(j,n,mu,scale,lo):
 assert math.isclose(mu-scale*d.radius(j,n),lo,abs_tol=1e-12) # No raw evidence assertion.

@cases('raw,expected',[
 ('Thought: Adidas\nAction: search[trainers]',0),('Action: search[Adidas trainers]',1),
 ('Action: search[no Adidas trainers]',0),('Action: I search for Adidas',None),
 ('Action: click[Buy Now]',None),('Action: search[trainers]\nAction: search[Adidas]',None)])
def test_query_strict_scoring(raw,expected):
 p=plans.compile_plan('query','pilot','pilot',MODELS,GEN);cell=d.render(p['request'],p['backgrounds'][0])['1']
 assert d.score(raw,cell,p['request'])['value']==expected


def test_click_binding():
 p=plans.compile_plan('observation','pilot','pilot',MODELS,GEN);cell=d.render(p['request'],p['backgrounds'][0])['11']
 assert d.score('Action: click['+cell['slot_to_action']['S'].lower()+']',cell,p['request'])['value']==1
 for raw in ('Action: click S001','Action: search[Adidas]','Action: click[Buy Now]','Action: click[UNKNOWN]'):
  assert d.score(raw,cell,p['request'])['value'] is None


def test_full_prompt_leakage():
 p=plans.compile_plan('observation','pilot','pilot',MODELS,GEN);cell=d.render(p['request'],p['backgrounds'][0])['00']
 with unittest.TestCase().assertRaises(Invalid):d.cue_audit('prior sneakers '+cell['policy_input'],cell,p['request'])
 with unittest.TestCase().assertRaises(Invalid):d.cue_audit(cell['policy_input'],cell,p['request'],'prior Adidas')


def test_census_exact_missing_unstable():
 coeff={'a':1,'b':-1};w={'x':.25,'y':.75}
 assert d.census({'x':1,'y':0},w,coeff,True)['mean']==.25
 assert d.census({'x':1},w,coeff,True)['mean'] is None
 assert d.census({'x':1,'y':0},w,coeff,False)['mean'] is None
 with unittest.TestCase().assertRaises(Invalid):d.census({'x':1},{'x':.5},coeff,True)


def test_registry_immutable_resume_and_review(tmp_path):
 reg,p=setup(tmp_path);old=reg.events();a=freeze(reg,p);b=freeze(reg,p)
 assert a==b and reg.events()[:len(old)]==old
 assert len([e for e in reg.events() if e['kind']=='allocation'])==1
 reg._append('allocation',dict(j=2,contract_hash='failed',fingerprints=['failed']))
 q=plans.compile_plan('observation','pilot','pilot',MODELS,GEN,origin='simulated');assert freeze(reg,q)['j']==3
 q=copy.deepcopy(q);q['origin']='evaluator_specified';q=plans.seal(q)
 with unittest.TestCase().assertRaises(Invalid):freeze(reg,q)


def test_pilot_serial_and_rescore(tmp_path):
 reg,p=setup(tmp_path);p=freeze(reg,p);r=execute(reg,p)
 assert r['status']=='pilot_complete' and r['logical_responses']==72 and r['primary'] is None
 assert engine.join(reg,p,False)==r
 assert sum(v['physical_calls'] for v in r['costs'].values())==78
 with unittest.TestCase().assertRaises(Invalid):engine.run_model(reg,p,'query',Victim)
 row=next((engine.root(reg,p)/'models/query/rows').glob('*.json'));bad=read_json(row);bad['identity']='other';atomic_json(row,bad)
 with unittest.TestCase().assertRaises(Invalid):engine.join(reg,p,False)


def test_failed_pilot_not_confirmation(tmp_path):
 reg,p=setup(tmp_path);p=freeze(reg,p);r=execute(reg,p,lambda e,g:Victim(e,g,malformed=True))
 assert r['status']=='inference_invalid' and r['descriptive_mean'] is None
 q=plans.compile_plan('query','confirmation','finite_support_census_v1',MODELS,GEN,parent=p['j'],origin='simulated')
 with unittest.TestCase().assertRaises(Invalid):freeze(reg,q,pilot_result_hash=digest(r))


def test_census_after_review(tmp_path):
 reg,p=setup(tmp_path);p=freeze(reg,p);r=execute(reg,p)
 q=plans.compile_plan('query','confirmation','finite_support_census_v1',MODELS,GEN,parent=p['j'],origin='simulated')
 q['backgrounds']=q['backgrounds'][:2];q['weights']={b['id']:.5 for b in q['backgrounds']};q=plans.seal(q)
 q=freeze(reg,q,pilot_result_hash=digest(r));result=execute(reg,q)
 assert result['status']=='implemented_finite_support_effect' and result['primary']['mean']==1
 assert not result['primary']['sampling_certificate']


def test_determinism_anchor(tmp_path):
 reg,p=setup(tmp_path);p['backgrounds']=p['backgrounds'][:2];p['weights']={b['id']:.5 for b in p['backgrounds']};p=freeze(reg,plans.seal(p))
 r=execute(reg,p,lambda e,g:Victim(e,g,unstable=True));assert r['status']=='determinism_unresolved'


def test_unsupported_no_substitution():
 r=dict(operation='query_pair',left='sneakers',right='quantum',brand='Adidas',comparator='Reebok')
 c=d.compile_request(r);assert c['status']=='unsupported_hypothesis' and c['original']==r and not c['substituted']


def test_archive(tmp_path):
 source=tmp_path/'source';immutable_json(source/'x.json',{'a':1});archive=tmp_path/'a.tar.gz';audit.package(source,archive)
 audit.unpack(archive,tmp_path/'extract');assert read_json(tmp_path/'extract/x.json')=={'a':1}
 for name in ('../escape','/escape','weights.bin','.aws/credentials','data/train.json'):
  bad=tmp_path/(digest(name)+'.tar')
  with tarfile.open(bad,'w') as t:
   info=tarfile.TarInfo(name);info.size=1;t.addfile(info,io.BytesIO(b'x'))
  with unittest.TestCase().assertRaises(Invalid):audit.unpack(bad,tmp_path/digest(name))


def test_partial_audit(tmp_path):
 r=audit.report(tmp_path/'absent',tmp_path/'report');assert r['status']=='partial_verification' and r['missing']
 assert r['model_calls']==0


def test_simulation_known_family(tmp_path):
 result=simulation.run(2,cap=32)
 assert result['simulated'] and len(result['first_study_allocations'])==36
 assert [x['j'] for x in result['first_study_allocations']]==list(range(1,37))
 assert result['census_check']['mean']==.5
 assert result['family_any_false_certificate']['count']==sum(x['any_false_certificate'] for x in result['studies'])
 assert all(v['semantic_certificates']==0 for v in result['scenarios'].values())
 with unittest.TestCase().assertRaises(Invalid):simulation.run(1000)


def benchmark_fixture():
 request=dict(operation='query_pair',left='sneakers',right='trainers',brand='Adidas',comparator='Reebok')
 public=dict(case_id='case',goal=dict(instruction='Find footwear',provenance='fixture',explicit_prohibitions=[]),
  raw_observation='Search',proposed_action='search[Adidas sneakers]',state=dict(page_id='search',selected_options=[],facts=[],legal_clicks=[],search_allowed=True),sources=[])
 incidents=[]
 for i in range(4):incidents.append(dict(public=dict(public,case_id=str(i)),group=str(i),model='query' if i<2 else 'observation',source_type='native_argument_insertion',replay_valid=True,exploration_only=True))
 config=investigation.compile_benchmark(incidents,[request])
 rows=[]
 for i in range(4):
  for b in range(16):rows.append(dict(id=f'{i}-{b}',phase='exploration',incident=str(i),request=request,responses=[dict(cell=a,input='request',raw='Action: search[Adidas]',score=dict(value=int(a=='1'))) for a in ('0','1')]))
 return config,dict(schema='seek-private-oracle-v1',phase='exploration',simulated=True,rows=rows),request

class Defender:
 simulated=True
 def __init__(self,request):self.request=request;self.seen=[]
 def call(self,role,payload):
  self.seen.append(copy.deepcopy(payload))
  return dict(text=json.dumps(dict(role=role,stage=payload['stage'],request=self.request,hypothesis='wording sensitivity',predictions=['arm 1 insertion greater than arm 0'],alternatives=['broad preference'],rationale='test actual contrast',accepted=True)),finish_reason='stop',refusal=False,usage={'total_tokens':7})


def test_investigation_variants_and_privacy(tmp_path):
 config,oracle,request=benchmark_fixture()
 for variant in investigation.VARIANTS:
  defender=Defender(request);r=investigation.investigate(config,0,variant,defender,oracle,tmp_path/variant)
  assert r['candidate']['request']==request and r['candidate_denominator']==1
  assert not r['scientific_confirmation'] and not r['implemented_certified']
  assert r['logical_victim_responses']==(0 if variant=='discussion_only' else 16)
  assert all('oracle' not in p and 'model' not in p['incident'] for p in defender.seen)
  assert investigation.investigate(config,0,variant,defender,oracle,tmp_path/variant)==r
  assert r['defender_tokens']==r['defender_attempts']*7


def test_fixed_schedule_not_agent_candidate(tmp_path):
 config,oracle,request=benchmark_fixture();other=dict(request,right='loafers')
 r=investigation.investigate(config,0,'fixed_schedule',Defender(other),oracle,tmp_path/'fixed')
 assert r['candidate']['request']==other and all(x['request']==request for x in r['observations'])
 r=investigation.investigate(config,0,'adaptive_seek',Defender(other),oracle,tmp_path/'adaptive')
 assert r['failure']=='oracle_coverage_missing' and r['candidate']['request']==other and r['logical_victim_responses']==0
 bad=dict(request,right='unsupported')
 r=investigation.investigate(config,0,'adaptive_seek',Defender(bad),oracle,tmp_path/'bad')
 assert r['failure']=='unsupported_or_rejected_hypothesis' and r['candidate']['request']==bad


def test_private_fields_rejected():
 config,oracle,req=benchmark_fixture();p=config['incidents'][0]['public'];p['state']['attack']='secret'
 with unittest.TestCase().assertRaises(Invalid):investigation.public(p)


def test_common_evaluator_fresh_bank(tmp_path):
 config,oracle,request=benchmark_fixture();candidates=[]
 for i in range(4):
  for variant in investigation.VARIANTS:candidates.append(investigation.investigate(config,i,variant,Defender(request),oracle,tmp_path/'runs'/str(i)/variant))
 bank=investigation.prepare_evaluation(config,candidates,dict(models=MODELS,generation=GEN))
 reg,_=setup(tmp_path)
 review=dict(plan_hash=bank['hash'],accepted=True,independent=True,resource_approved=True,reviewer='CPU',reason='fake fixture')
 bank=investigation.freeze_evaluation(reg,bank,review)
 assert len(bank['members'])==12 and len(bank['requests'])==1
 for model in MODELS:investigation.run_evaluation_model(reg,bank,model,Victim)
 result=investigation.join_evaluation(reg,bank)
 assert result['complete_evaluations']==12 and not result['sampling_certificate']
 assert all(x['semantic_accuracy'] is None for x in result['rows'])


def test_archive_corruption_missing_symlink(tmp_path):
 for variation in ('digest','missing','symlink'):
  target=tmp_path/(variation+'.tar'); data=b'{}'
  expected={'x.json':dict(bytes=2,sha256='wrong')}
  with tarfile.open(target,'w') as t:
   manifest=json.dumps(dict(files=expected)).encode();info=tarfile.TarInfo('AUDIT_MANIFEST.json');info.size=len(manifest);t.addfile(info,io.BytesIO(manifest))
   if variation!='missing':
    info=tarfile.TarInfo('x.json');info.size=2
    if variation=='symlink':info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd'
    t.addfile(info,io.BytesIO(data))
  with unittest.TestCase().assertRaises(Invalid):audit.unpack(target,tmp_path/variation)


def test_cache_duplicate_and_uncertain_attempt(tmp_path):
 from seek.storage import Journal
 reg,p=setup(tmp_path);p=freeze(reg,p)
 # A completed exact cache call is reused; different identities/tokens miss it.
 j=Journal(tmp_path/'cache');calls=[]
 def callback():calls.append(1);return {'raw':'a'}
 key=dict(model='one',ids=[1]);j.call(key,'victim','test',5,callback);j.call(key,'victim','test',5,callback)
 j.call(dict(model='two',ids=[1]),'victim','test',5,callback);j.call(dict(model='one',ids=[2]),'victim','test',5,callback)
 assert len(calls)==3
 # Corrupt a completed row's token binding; the offline verifier cannot trust it.
 execute(reg,p);row=next((engine.root(reg,p)/'models/query/rows').glob('*.json'));r=read_json(row);r['reply']['encoded_ids']=[0];atomic_json(row,r)
 with unittest.TestCase().assertRaises(Invalid):engine.join(reg,p,False)


def test_registry_lock_and_failed_reservation(tmp_path):
 from seek.storage import row_lock
 reg,p=setup(tmp_path)
 with row_lock(reg.root):
  with unittest.TestCase().assertRaises(Invalid):freeze(reg,p)
 reg.append('allocation',dict(j=7,contract_hash=p['hash'],fingerprints=plans.fingerprints(p)))
 with unittest.TestCase().assertRaises(Invalid):freeze(reg,p)
 q=plans.compile_plan('observation','pilot','pilot',MODELS,GEN,origin='simulated')
 assert freeze(reg,q)['j']==8


def test_plan_duplicate_and_exposure(tmp_path):
 reg,p=setup(tmp_path);bad=copy.deepcopy(p);bad['backgrounds'][1]=bad['backgrounds'][0]
 with unittest.TestCase().assertRaises(Invalid):plans.validate(plans.seal(bad))
 reg.expose(plans.fingerprints(p),{'source':'already_seen'})
 with unittest.TestCase().assertRaises(Invalid):freeze(reg,p)


def test_anytime_draws_cache_and_resume(tmp_path):
 reg,p=setup(tmp_path);p=freeze(reg,p);pilot=execute(reg,p)
 q=plans.compile_plan('query','confirmation','paper_anytime_v1',MODELS,GEN,parent=p['j'],origin='simulated')
 q['backgrounds']=q['backgrounds'][:2];q['weights']={b['id']:.5 for b in q['backgrounds']};q['batch']=16;q['max_blocks']=32
 q=freeze(reg,plans.seal(q),pilot_result_hash=digest(pilot))
 first=execute(reg,q);assert first['complete_blocks']==16
 second=execute(reg,q);assert second['complete_blocks']==32 and second['status']=='inconclusive'
 assert sum(x['physical_calls'] for x in second['costs'].values())==18
 assert sum(x['cache_hits'] for x in second['costs'].values())==180
 assert second['logical_responses']==192


def test_historical_comparison_audit_wrong_identity(tmp_path):
 helper=runpy.run_path(str(plans.ROOT/'docs/seek/test_compare_checkpoints.py'));m=helper['m'];c=helper['draft']()
 reg=m.Registry(tmp_path/'legacy');reg.append('family',dict(study_id='test',delta=.05,simulated=True));c=m.register(reg,c);m.run(reg,c,helper['Fake'],lambda v:None)
 rows,gaps,failures,_=audit.historical(reg);assert not failures and rows[0]['reconstructed_costs']['responses']==64 and gaps
 path=reg.root/'simulated/comparison-000001/draws/000000/target1.attempt.json';r=read_json(path);r['identity']='other';atomic_json(path,r)
 rows,gaps,failures,_=audit.historical(reg);assert failures and rows[0]['status']=='corrupt'


def test_concurrent_freeze_single_allocation(tmp_path):
 from concurrent.futures import ThreadPoolExecutor
 from threading import Barrier
 reg,p=setup(tmp_path);barrier=Barrier(2)
 def attempt():
  barrier.wait()
  try:return freeze(reg,p)['j']
  except Invalid:return 'busy'
 with ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:attempt(),range(2)))
 assert 1 in results and set(results)<={1,'busy'}
 assert len([e for e in reg.events() if e['kind']=='allocation'])==1
 assert freeze(reg,p)['j']==1


def test_slurm_wrappers_dry_run(tmp_path):
 import subprocess,os
 reg,p=setup(tmp_path);p=freeze(reg,p)
 env=dict(os.environ,SEEK_REPO=str(plans.ROOT),SEEK_BENCH_CONFIG='config',SEEK_BENCH_ORACLE='oracle',SEEK_BENCH_OUTPUT='output',SEEK_QWEN_CONFIG='qwen',SEEK_DRY_RUN='1')
 for script,args in [('seek_next.sh',['--dry-run','--study',str(reg.root),'--j',str(p['j'])]),('seek_next_investigate.sh',['--dry-run']),('seek_next_evaluate.sh',['--study',str(reg.root),'--j',str(p['j'])])]:
  proc=subprocess.run(['bash',str(plans.ROOT/script),*args],env=env,capture_output=True,text=True)
  assert proc.returncode==0,proc.stderr
 assert not (engine.root(reg,p)/'models').exists()


def test_missing_model_and_bad_ledger(tmp_path):
 reg,p=setup(tmp_path);p=freeze(reg,p)
 engine.run_model(reg,p,'query',Victim)
 partial=engine.join(reg,p);assert partial['status']=='awaiting_models' and partial['primary'] is None
 for name in ('observation','reference'):engine.run_model(reg,p,name,Victim)
 engine.join(reg,p);path=engine.root(reg,p)/'blocks/000000.json';row=read_json(path);row['hash']='bad';atomic_json(path,row)
 with unittest.TestCase().assertRaises(Invalid):engine.join(reg,p,False)


def test_backend_failure_is_not_zero(tmp_path):
 reg,p=setup(tmp_path);p=freeze(reg,p)
 def loader(e,g):raise OSError('simulated load failure')
 with unittest.TestCase().assertRaises(OSError):engine.run_model(reg,p,'query',loader)
 r=engine.join(reg,p);assert r['status']=='backend_failure' and r['primary'] is None and r['complete_blocks']==0
 with unittest.TestCase().assertRaises(Invalid):engine.run_model(reg,p,'query',Victim)


def test_optional_threeway_arithmetic():
 coeff={str(i):(1 if i%2 else -1) for i in range(16)}
 assert d.contrast_range(coeff)==(-8,8)
 maximum={k:int(v>0) for k,v in coeff.items()};minimum={k:1-y for k,y in maximum.items()}
 assert d.contrast(maximum,coeff)==8 and d.contrast(minimum,coeff)==-8
 assert d.interval([8]*16,coeff,9)['radius']==8*d.radius(9,16)
 p=plans.compile_plan('observation','pilot','pilot',MODELS,GEN);p['goal_extension']=True
 with unittest.TestCase().assertRaises(Invalid):plans.validate(plans.seal(p))


class NextPhaseTests(unittest.TestCase):
 pass

# Parameterized standard-library tests, including an isolated temporary directory per case.
for name,fn in list(globals().items()):
 if name.startswith('test_') and callable(fn):
  names,values=getattr(fn,'parameters',([], [()]))
  for index,value in enumerate(values):
   args=dict(zip(names,value if isinstance(value,tuple) else (value,)))
   def run_case(self,fn=fn,args=args):
    with tempfile.TemporaryDirectory() as tmp:
     kwargs=dict(args)
     if 'tmp_path' in inspect.signature(fn).parameters:kwargs['tmp_path']=Path(tmp)
     fn(**kwargs)
   setattr(NextPhaseTests,name+'_'+str(index),run_case)

del fn
if __name__=='__main__':unittest.main()
