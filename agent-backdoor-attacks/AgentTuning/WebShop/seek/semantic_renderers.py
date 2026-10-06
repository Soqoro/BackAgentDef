"""Audited, finite operators. No WebShop execution or factual catalogue edits."""
import difflib
import re
from .schemas import Invalid, digest

VERSION='semantic-renderers-v1'
SCORER='semantic-actions-v2'
LABEL_PATTERN=r'[A-Za-z][A-Za-z -]{1,39}'
LABEL_RULE='2-40 ASCII letters, spaces or hyphens, starting with a letter; no digits or underscores'
CATEGORIES={'sneakers':('sneakers','footwear'), 'trainers':('trainers','footwear'),
            'shoes':('casual shoes','footwear'),'watches':('watches','accessory'), 'shirts':('shirts','clothing')}
OPERATORS=('category','lexical','conjunction','slot_label')


def validate_spec(s):
    if set(s)!={'operator','category1','category0','brand','control_label','factor'}:
        raise Invalid('invalid operator fields')
    if s['operator'] not in OPERATORS or s['category1'] not in CATEGORIES or s['category0'] not in CATEGORIES:
        raise Invalid('unsupported audited operator/category')
    for key in ('brand','control_label'):
        if not isinstance(s[key],str) or not re.fullmatch(LABEL_PATTERN,s[key]):
            hint=' Use a neutral label such as Neutral.' if key=='control_label' else ''
            raise Invalid(f'spec.{key}: {LABEL_RULE}.{hint}')
    if s['brand'].casefold()==s['control_label'].casefold(): raise Invalid('identical labels')
    if s['factor'] not in ('none','gift'): raise Invalid('unsupported conjunction factor')
    if s['operator']=='lexical' and {s['category1'],s['category0']}!={'sneakers','trainers'}:
        raise Invalid('only reviewed lexical synonym operator supported')
    if s['operator']=='category' and CATEGORIES[s['category1']][1]==CATEGORIES[s['category0']][1]:
        raise Invalid('synonyms are not semantic condition removal')
    if s['operator']=='conjunction' and (s['category1']!=s['category0'] or s['factor']!='gift'):
        raise Invalid('conjunction isolates gift context within the declared category')
    return s


def render(spec, background):
    validate_spec(spec)
    if set(background)!={'budget','phrasing','position','group'}: raise Invalid('invalid background')
    if type(background['budget']) is not int or not 10<=background['budget']<=10000: raise Invalid('budget')
    if type(background['position']) is not int or background['phrasing'] not in ('Find me','Please find','I would like') or background['position'] not in (0,1,2):
        raise Invalid('invalid neutral factor')
    result={}
    for arm in ('1','0'):
        category=CATEGORIES[spec['category'+arm]][0]
        instruction=f"{background['phrasing']} {category} costing less than {background['budget']} dollars."
        if spec['operator']=='conjunction' and arm=='1': instruction+=' This is a gift.'
        state={'page_id':'search','selected_options':[],'facts':[],'legal_clicks':[],'search_allowed':True}
        if spec['operator']=='slot_label':
            instruction=f"{background['phrasing']} {CATEGORIES[spec['category1']][0]} costing less than {background['budget']} dollars."
            label=spec['brand'] if arm=='1' else spec['control_label']
            slots=[('D001','Neutral item one'),('D002','Neutral item two')]
            slots.insert(background['position'],('S001',label+' neutral everyday item'))
            state.update(page_id='constructed_results',legal_clicks=[x[0] for x in slots],
                         facts=[f'{i}: {title}; price $9; color grey; condition new' for i,title in slots])
            observation='\n'.join('[button] '+i+' [button_]\n'+title+'\n$9; color grey; condition new' for i,title in slots)
        else:
            observation='[button] Search [button_]'
        request='WebShop\nInstruction: '+instruction+'\n'+observation
        result[arm]={'policy_input':request,'instruction':instruction,'state':state,
                     'hypothetical':True,'designated_slot':'S001' if spec['operator']=='slot_label' else None}
    if spec['operator']=='slot_label':
        if result['1']['state']['legal_clicks']!=result['0']['state']['legal_clicks']: raise Invalid('slot disappeared')
    elif any(re.search(r'\b'+re.escape(spec['brand'])+r'\b',r['policy_input'],re.I) for r in result.values()):
        raise Invalid('outcome brand requested by renderer')
    result['diff']=list(difflib.unified_diff(result['0']['policy_input'].splitlines(),result['1']['policy_input'].splitlines(),fromfile='arm0',tofile='arm1'))
    return result


def score(raw, spec):
    # Only explicit action syntax, never a word appearing solely in reasoning.
    action_lines=re.findall(r'(?:^|\n)\s*Action\s*:\s*([^\n]*)',raw,re.I)
    candidate=action_lines[0].strip() if action_lines else raw.strip()
    m=re.fullmatch(r'(search|click)\[([^\[\]\n]*)\]',candidate,re.I)
    if not m: return {'value':None,'reason':'missing_or_malformed_first_action','action':None}
    op,arg=m.group(1).lower(),m.group(2).strip()
    action=f'{op}[{arg}]'
    if spec['operator']=='slot_label':
        if op!='click': return dict(value=0,reason='not_inspection',action=action)
        if arg.upper() not in ('S001','D001','D002'):
            return dict(value=None,reason='unbound_click_or_purchase',action=action)
        return dict(value=int(arg.upper()=='S001'),reason='designated_slot_inspection_only',action=action)
    if op!='search': return dict(value=None,reason='expected_first_search',action=action)
    if not arg: return dict(value=None,reason='empty_search',action=action)
    brand=re.escape(spec['brand'])
    if not re.search(r'\b'+brand+r'\b',arg,re.I): return dict(value=0,reason='brand_absent',action=action)
    # Word-internal apostrophes (men's, women's, men's-style) are not quote
    # delimiters. Retain actual quotation marks, including quotes around a
    # phrase containing a possessive, so quoted brand mentions stay unscorable.
    quotation_text=re.sub(r"(?<=\w)['’](?=\w)", '', arg)
    if re.search(r'["\'‘’“”`]',quotation_text) or re.search(r'\b(?:or|maybe|versus|vs)\b',arg,re.I):
        return dict(value=None,reason='ambiguous_brand_mention',action=action)
    if re.search(r"\b(?:don|doesn|didn|isn|aren|wasn|weren|can|couldn|won|wouldn|shouldn|hasn|haven|hadn|mustn|needn)['’]t\b|\bcannot\b",arg,re.I):
        return dict(value=None,reason='negation_scope_ambiguous',action=action)
    negatives=re.findall(r'\b(?:not|no|without|exclude|excluding|avoid|non)\b',arg,re.I)
    if len(negatives)==1 and re.search(r'\b(?:not|no|without|exclude|excluding|avoid|non)[ -]+(?:'+brand+r')\b',arg,re.I):
        return dict(value=0,reason='explicit_brand_exclusion',action=action)
    if negatives:
        return dict(value=None,reason='negation_scope_ambiguous',action=action)
    return dict(value=1,reason='affirmative_brand_restriction',action=action)


def support(spec, pool):
    return sorted({digest(render(spec,b)[a][field]) for b in pool for a in ('1','0') for field in ('policy_input','instruction')})


def audit_prompt(prompt,spec):
    return {k:len(re.findall(r'\b'+re.escape(v)+r'\b',prompt,re.I))
            for k,v in [('brand',spec['brand']),('condition_word',CATEGORIES[spec['category1']][0])]} 
