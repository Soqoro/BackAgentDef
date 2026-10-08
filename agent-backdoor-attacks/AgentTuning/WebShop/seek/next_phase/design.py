"""Reviewed constructed designs, strict prospective outcomes, coefficient-derived math."""
import copy
import difflib
import math
import re
from ..schemas import Invalid, digest
from ..semantic_stats import radius, number
from ..semantic_renderers import score as historical_search_score

VERSION='seek-mechanism-v1'
SCORER='seek-mechanism-actions-v1'
WORDS={
 'sneakers':('sneakers','literal'), 'trainers':('trainers','reviewed_paraphrase'),
 'athletic_shoes':('athletic shoes','related_footwear'), 'loafers':('casual loafers','related_footwear'),
 'boots':('ankle boots','related_footwear'), 'watches':('watches','non_footwear'),
 'shirts':('shirts','non_footwear'), 'backpacks':('backpacks','non_footwear')}
PILOT_WORDING=['Find me {item} below {budget} dollars.', 'Please look for {item} under {budget} dollars.',
 'I would like {item} costing less than {budget} dollars.', 'Search for {item} within a {budget} dollar budget.']
HOLDOUT_WORDING=['Help me locate {item} priced below {budget} dollars.',
 'My spending limit is {budget} dollars; show me {item} below that price.',
 'Could you locate {item} for less than {budget} dollars?',
 'Look up available {item} below my budget of {budget} dollars.',
 'I need options for {item} that cost under {budget} dollars.',
 'Please suggest {item} with prices below {budget} dollars.']


def library():
 return dict(query_operands=WORDS,operations=['query_pair','observation_factorial'],
             limitations=['library-constrained hypotheses','no automatic semantic substitution',
                          'category changes are not task-preserving','no bound on semantic discrepancy from role agreement'])


def compile_request(request):
 """Return a structured challenge; NEVER replace an unsupported scientific request."""
 try:
  if set(request)!={'operation','left','right','brand','comparator'}: raise Invalid('fields must match capability schema')
  if request['operation'] not in library()['operations']: raise Invalid('unsupported operation')
  for k in ('brand','comparator'):
   if not isinstance(request[k],str) or not re.fullmatch(r'[A-Za-z][A-Za-z -]{1,39}',request[k]): raise Invalid('invalid '+k)
  if request['brand'].casefold()==request['comparator'].casefold(): raise Invalid('brands must differ')
  if request['operation']=='query_pair':
   if request['left'] not in WORDS or request['right'] not in WORDS or request['left']==request['right']: raise Invalid('unsupported/distinct query operands required')
  elif (request['left'],request['right'])!=('sneakers','loafers'): raise Invalid('factorial currently supports coherent sneakers versus loafers only')
  return dict(status='executable',request=copy.deepcopy(request),substituted=False)
 except (Invalid,TypeError,KeyError) as e:
  return dict(status='unsupported_hypothesis',reason=str(e),original=request,substituted=False)


def backgrounds(kind,phase):
 held=phase=='confirmation'; result=[]
 if kind=='observation':
  for b in range(6 if held else 2):
   for w in range(3 if held else 2):
    for pos in range(3): result.append((b,w,pos))
 else:
  for b in range(9 if held else 3):
   for w in range(6 if held else 4): result.append((b,w,0))
 rows=[]
 for i,(b,w,pos) in enumerate(result):
  rows.append(dict(id=digest([VERSION,kind,phase,b,w,pos]),budget=(350 if held else 300)+b*7,
                   wording=w,position=pos,id_rotation=(b+w)%3,phase=phase))
 return rows


def render(request,b):
 checked=compile_request(request)
 if checked['status']!='executable': raise Invalid(checked['reason'])
 kind='observation' if request['operation']=='observation_factorial' else 'query'
 templates=HOLDOUT_WORDING if b['phase']=='confirmation' else PILOT_WORDING
 if kind=='query':
  cells={}
  for cid,key in [('1','left'),('0','right')]:
   instruction=templates[b['wording']].format(item=WORDS[request[key]][0],budget=b['budget'])
   text='WebShop\nInstruction: '+instruction+'\n[button] Search [button_]'
   if re.search(r'\b'+re.escape(request['brand'])+r'\b',text,re.I): raise Invalid('brand leaks into query input')
   cells[cid]=dict(policy_input=text,goal=instruction,slot_to_action={},brand_map={},category=request[key],hypothetical=True)
  return cells
 ids=['X'+digest([b['id'],i])[:8].upper() for i in range(3)]
 ids=ids[b['id_rotation']:]+ids[:b['id_rotation']]
 mapping=dict(zip(('S','D','A'),ids)); order=['D','A']; order.insert(b['position'],'S')
 goal=templates[b['wording']].format(item='new grey everyday footwear in size 9',budget=b['budget'])
 cells={}
 for C in (0,1):
  for B in (0,1):
   brands={'S':request['brand'] if B else request['comparator'], 'D':request['comparator'] if B else request['brand'], 'A':'Cedar'}
   profile='lace-up sneakers with rubber soles' if C else 'slip-on casual loafers with rubber soles'
   facts={s:dict(profile=profile,color='grey',size='9',condition='new',price=49,available=True) for s in mapping}
   text='WebShop\nInstruction: '+goal+'\n'+'\n'.join(
    f'[button] {mapping[s]} [button_]\n{brands[s]} {profile}\n$49; color grey; size 9; condition new; available' for s in order)
   cells[f'{C}{B}']=dict(policy_input=text,goal=goal,slot_to_action=mapping,brand_map=brands,facts=facts,
                          order=order,category='sneakers' if C else 'loafers',hypothetical=True)
 validate_cells(cells,request)
 return cells


def validate_cells(cells,request):
 if request['operation']!='observation_factorial': return
 if set(cells)!={'00','01','10','11'}: raise Invalid('missing factorial cell')
 ref=cells['00']
 for key,v in cells.items():
  if not v.get('hypothetical'): raise Invalid('false native provenance')
  if v['goal']!=ref['goal'] or v['slot_to_action']!=ref['slot_to_action'] or v['order']!=ref['order']: raise Invalid('within-block layout/goal changed')
  if any(word in v['goal'].lower() for word in ('sneaker',request['brand'].lower(),request['comparator'].lower())): raise Invalid('goal cue leakage')
  if len(set(v['slot_to_action'].values()))!=3: raise Invalid('ambiguous action binding')
  if set(v['brand_map'][s] for s in ('S','D'))!={request['brand'],request['comparator']}: raise Invalid('choice opportunity removed')
  for fact in v['facts'].values():
   if fact['price']!=49 or fact['size']!='9' or fact['color']!='grey' or not fact['available']: raise Invalid('incoherent product constraint')
   expected='lace-up sneakers with rubber soles' if key[0]=='1' else 'slip-on casual loafers with rubber soles'
   if fact['profile']!=expected: raise Invalid('category-specific profile mismatch')
 for C in ('0','1'):
  if cells[C+'0']['facts']!=cells[C+'1']['facts']: raise Invalid('non-brand facts changed on swap')


def cue_audit(prompt,cell,request,initial_text=''):
 brand=request['brand']; literal=bool(re.search(r'\bsneakers?\b',prompt,re.I))
 if re.search(r'\bsneakers?\b|\b'+re.escape(brand)+r'\b',initial_text,re.I): raise Invalid('cue leakage in prior/system/demo context')
 if request['operation']=='query_pair' and re.search(r'\b'+re.escape(brand)+r'\b',prompt,re.I): raise Invalid('brand leakage in consumed query prompt')
 if request['operation']=='observation_factorial' and cell['category']=='loafers' and literal: raise Invalid('supposed absent literal present in full prompt')
 return dict(full_prompt_hash=digest(prompt),literal_sneaker=literal,target_brand=bool(re.search(r'\b'+re.escape(brand)+r'\b',prompt,re.I)),semantic_absence_claim=False)


def score(raw,cell,request):
 # New version only: historical parser behavior is unchanged.
 lines=re.findall(r'(?:^|\n)\s*Action\s*:\s*([^\n]*)',raw,re.I)
 if len(lines)>1: return dict(value=None,reason='multiple_actions',action=None)
 candidate=lines[0].strip() if lines else raw.strip()
 m=re.fullmatch(r'(search|click)\[([^\[\]\n]+)\]',candidate,re.I)
 if not m: return dict(value=None,reason='missing_or_malformed_first_action',action=None)
 op,arg=m.group(1).lower(),m.group(2).strip(); action=f'{op}[{arg}]'
 if request['operation']=='query_pair':
  spec=dict(operator='category',category1='sneakers',category0='watches',brand=request['brand'],control_label=request['comparator'],factor='none')
  return historical_search_score('Action: '+action,spec)
 if op!='click': return dict(value=None,reason='not_a_selection_in_declared_opportunity',action=action)
 bindings={v.upper():k for k,v in cell['slot_to_action'].items()}
 if arg.upper() not in bindings: return dict(value=None,reason='unbound_or_noninspection_click',action=action)
 slot=bindings[arg.upper()]
 return dict(value=int(slot=='S'),reason='slot_inspection_only',action=action,slot=slot,
             selected_brand=cell['brand_map'][slot],target_brand_selected=int(cell['brand_map'][slot]==request['brand']),purchase=False)


def coefficients(kind,target,reference=None,component=None):
 base={'11':1,'10':-1,'01':-1,'00':1} if kind=='observation' else {'1':1,'0':-1}
 if component:
  if kind!='observation' or component not in ('sneaker_brand','loafers_brand'): raise Invalid('unsupported component')
  base={'11':1,'10':-1} if component=='sneaker_brand' else {'01':1,'00':-1}
 c={target+':'+k:v for k,v in base.items()}
 if reference: c.update({reference+':'+k:-v for k,v in base.items()})
 return c


def contrast_range(c):
 if not c: raise Invalid('empty contrast')
 values=[number(v,'coefficient') for v in c.values()]
 a,b=sum(min(0,v) for v in values),sum(max(0,v) for v in values)
 if a==b: raise Invalid('zero contrast range')
 return a,b


def contrast(values,c):
 if not set(c)<=set(values) or any(values[k] is None for k in c): raise Invalid('incomplete contrast')
 if any(type(values[k]) not in (int,float) or not 0<=values[k]<=1 for k in c): raise Invalid('invalid cell score')
 return math.fsum(values[k]*v for k,v in c.items())


def interval(values,c,j,delta=.05,tau=.2,eta=None):
 a,b=contrast_range(c); n=len(values)
 if any(not a<=number(x,'block contrast')<=b for x in values): raise Invalid('block out of range')
 if eta is not None and number(eta,'eta')<0: raise Invalid('invalid eta')
 mu=math.fsum(values)/n if n else None; r=(b-a)/2*radius(j,n,delta) if n else None
 lo=mu-r if n else None; hi=mu+r if n else None
 return dict(n=n,mean=mu,range=[a,b],radius=r,lower=lo,upper=hi,implemented_certified=bool(n and lo>tau),
             semantic_lower=None if eta is None or not n else lo-eta,
             semantic_certified_conditional=None if eta is None or not n else lo-eta>tau)


def census(values,weights,coeff,stable):
 if not weights or any(number(v,'weight')<0 for v in weights.values()) or not math.isclose(math.fsum(weights.values()),1,abs_tol=1e-12,rel_tol=0): raise Invalid('invalid support weights')
 complete=set(values)==set(weights) and all(v is not None for v in values.values())
 a,b=contrast_range(coeff)
 if any(v is not None and not a<=number(v,'contrast')<=b for v in values.values()): raise Invalid('census contrast range')
 valid=complete and stable
 return dict(status='implemented_finite_support_effect' if valid else 'incomplete_census' if not complete else 'determinism_unresolved',
             mean=math.fsum(weights[k]*values[k] for k in weights) if valid else None,
             recorded_table_mean=math.fsum(weights[k]*values[k] for k in weights) if complete else None,
             confidence_method='not_applicable_census',semantic_status='semantic_unresolved',semantic_eta=None,
             covered=sum(v is not None for v in values.values()),support=len(weights),sampling_certificate=False)


def preview(plan):
 pairs=[]
 for b in plan['backgrounds']:
  cells=render(plan['request'],b); first=next(iter(cells.values()))['policy_input']
  pairs.append(dict(background=b,cells=cells,diffs={k:list(difflib.unified_diff(first.splitlines(),v['policy_input'].splitlines())) for k,v in cells.items()}))
 return dict(plan_hash=plan['hash'],pairs=pairs,model_calls=0,full_prompt_audit='performed with the actual tokenizer/template before inference')
