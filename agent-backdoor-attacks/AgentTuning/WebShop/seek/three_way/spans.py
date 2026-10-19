"""Public, offset-bound arbitrary span capabilities for the existing HACS Search.

Literal strings are data, never regexes or evaluator truth. Edits require a
reviewed incidental-field boundary; candidate extraction scans every public field.
"""
import copy
import re
from ..schemas import Invalid, digest
from ..discovery_phase import hacs
from ..discovery_phase.common import seal

VERSION='seek-public-span-v1'
PUBLIC_FIELDS=('policy_input','query','observation')


def literal(value):
    if not isinstance(value,str) or not 1<=len(value)<=512 or '\x00' in value:
        raise Invalid('bounded_literal_required')
    return value


def candidates(public, cap=64, max_chars=16384):
    if type(cap) is not int or not 1<=cap<=256: raise Invalid('span_cap')
    pools=[]; unscanned={}; total=0
    for field in PUBLIC_FIELDS:
        if field not in public: continue
        text=public[field]
        if not isinstance(text,str): raise Invalid('public_text_required')
        prefix=text[:max_chars];unscanned[field]=max(0,len(text)-max_chars)
        tokens=list(re.finditer(r'\S+',prefix)); spans=[];seen=set()
        # Round-robin across scales and fields rather than privileging a note.
        scales=[]
        for width in (1,2,4,8):
            scales.append([(tokens[i].start(), tokens[i+width-1].end()) for i in range(max(0,len(tokens)-width+1))])
        scales.append([(m.start(),m.end()) for m in re.finditer(r'[^\n]+',prefix)])
        for i in range(max([len(x) for x in scales]+[0])):
            for scale in scales:
                if i>=len(scale):continue
                a,b=scale[i]
                if b-a>512 or (a,b) in seen:continue
                seen.add((a,b));spans.append(dict(field=field,span=[a,b],literal=text[a:b],source_hash=digest(text),
                    origin='public_offset_segmentation',matching='case_sensitive_substring',normalization='none'))
        pools.append(spans); total+=len(spans)
    rows=[]
    for i in range(max([len(p) for p in pools]+[0])):
        for pool in pools:
            if i<len(pool): rows.append(pool[i])
    for row in rows:row['id']='s-'+digest(row)[:20]
    return dict(schema=VERSION,candidates=rows[:cap],candidate_cap=cap,omitted_candidates=max(0,total-cap),
                unscanned_characters=unscanned,coverage_complete=False,private_marker_used=False,
                refinement='explicit bounded substring refinement available; no automatic oracle expansion')


def refine(candidate, text, cap=32):
    validate_span(candidate,text)
    a,b=candidate['span']; n=b-a; windows=[]
    for width in sorted({1,2,4,8,max(1,n//2)}):
        for start in range(a,b-width+1):
            value=dict(candidate,span=[start,start+width],literal=text[start:start+width],origin='public_multiscale_refinement',parent=candidate['id'])
            value['id']='s-'+digest({k:v for k,v in value.items() if k!='id'})[:20];windows.append(value)
    return dict(candidates=windows[:cap],omitted_candidates=max(0,len(windows)-cap),parent=candidate['id'])


def validate_span(candidate, text):
    if candidate['source_hash']!=digest(text): raise Invalid('stale_span_source')
    a,b=candidate['span']
    if type(a) is not int or type(b) is not int or not 0<=a<b<=len(text) or text[a:b]!=candidate['literal']:
        raise Invalid('shifted_or_invalid_span')
    literal(candidate['literal'])


def edit(public, candidate, replacement, constraints):
    field=candidate['field'];text=public[field];validate_span(candidate,text)
    if replacement:literal(replacement)
    a,b=candidate['span']; review=constraints.get(field,{})
    if review.get('source_hash')!=digest(text) or not review.get('reviewer'):raise Invalid('protected_field_review_required')
    allowed=review.get('editable',[]);protected=review.get('protected',[])
    if not any(x<=a and b<=y for x,y in allowed) or any(a<y and b>x for x,y in protected):
        raise Invalid('protected_goal_or_action_field_edit')
    result=copy.deepcopy(public);result[field]=text[:a]+replacement+text[b:]
    return result


def predicate(ast, features, domain):
    if ast['op'] in ('literal_substring','literal_exact'):
        value=features.get(ast['field'])
        if value is None:return None
        term=literal(ast['value'])
        return term in value if ast['op']=='literal_substring' else term==value
    if ast['op']=='and':
        vals=[predicate(x,features,domain) for x in ast['args']]
        return False if False in vals else None if None in vals else True
    return hacs.evaluate(ast,features,domain)


def compile_library(public, constraints, domain, outcome_id, cap=32):
    """Same family/bank contracts as HACS; no new search or inference engine."""
    inventory=candidates(public,cap); old=hacs.hypotheses(domain,outcome_id,32)
    hypotheses=copy.deepcopy(old['hypotheses']); experiments=[];unsupported=[]
    for candidate in inventory['candidates']:
        field=candidate['field'];value=candidate['literal']
        for op in ('literal_substring','literal_exact'):
            ast=dict(op=op,field=field,value=value)
            hypotheses.append(dict(id='h-'+digest(ast)[:16],ast=ast,origin='public_span_grammar',outcome_id=outcome_id,version=1))
        # Neutral literal is controller-declared, not an evaluator marker/target.
        neutral='x'*len(value);near=value[:-1]+('x' if value[-1]!='x' else 'y')
        for operation,replacement in (('deletion',''),('neutral_replacement',neutral),('near_match',near)):
            try:
                changed=edit(public,candidate,replacement,constraints)
            except Invalid as exc:
                unsupported.append(dict(candidate=candidate['id'],operator=operation,reason=str(exc)));continue
            for name,arms in [(operation,[public,changed])]+([('matched_reinsertion',[public,changed])] if operation=='neutral_replacement' else []):
                # Both directions explicitly fixed. Reinsertion compares restored source vs matched decoy.
                e=dict(id='e-'+digest([candidate['id'],name])[:16],operator='span_'+name,version=1,
                    logical_cost=2,features=arms,inputs=arms,span=candidate,origin='algorithmic_public_span',
                    direction='original/restored minus intervention',interpretation='reviewed incidental span; not an attack attribution')
                experiments.append(e)
    # Category operations remain available through the existing bank and renderer.
    for e in hacs.bank(domain,old): experiments.append(e)
    for e in experiments:
        e['predictions']={h['id']:[None if (v:=predicate(h['ast'],f,domain)) is None else .9 if v else .1 for f in e['features']] for h in hypotheses}
    family=dict(old,hypotheses=hypotheses,origin='declared_categories_plus_public_spans',epoch=0)
    return dict(family=family,bank=experiments,inventory=inventory,unsupported=unsupported,
                coverage=hacs.coverage(experiments,family),semantic_eta=None,scientific_confirmation=False)


def review_packet(search, event):
    """Controller arithmetic, compact references, same schedule for active/fixed."""
    if event not in ('initialization','semantic_change','finalization'):raise Invalid('role_review_not_scheduled')
    return dict(event=event,observed=[dict(id=e['id'],values=[a['value'] for a in e['arms']],difference=e['difference']) for e in search.events],
                equivalence_classes=search.coverage['classes'],unknown=search.coverage['unknown_hypotheses'],
                selected=[dict(experiment=d['experiment'],origin=d['origin']) for d in search.selections],
                weights_are_not_posterior=True,semantic_eta=None,authorization_status='not_inferred')
