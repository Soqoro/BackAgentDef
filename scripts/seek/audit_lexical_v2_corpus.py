#!/usr/bin/env python3
"""Bounded saved-corpus candidate audit; no labels, source approval or model calls."""
import argparse
import json
import re
from pathlib import Path
from prepare_lexical_v2 import ROOT, validate_contract
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash

CORPUS='8b1165ac799dc4c2bd96e0a13a1c39a098fc73cbb9c8ab453026dc545ae1fed1'


def inspect_record(record,rule):
    decision=dict(id=record.get('id'),source_record_hash=digest(record),status='rejected',reasons=[])
    turns=record.get('conversations',[])
    # Only a saved initial search page. Do not extract requirements from demonstrations or later pages.
    found=[]
    for i,t in enumerate(turns):
        if t.get('from')!='human':continue
        match=re.fullmatch(r'WebShop\s*\[SEP\]\s*Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*Search\s*',t.get('value',''),re.S)
        if match:found.append((i,match))
    if not found:
        decision['reasons']=['initial_search_page_not_found'];return decision
    i,match=found[0];goal=match.group(1)
    decision.update(human_turn=i,instruction=goal,instruction_span=list(match.span(1)),
        human_text_hash=digest(turns[i]['value']),later_turns_used=False)
    # Deliberately narrow grammar: reject unstructured color mentions and alternative values.
    colors=re.findall(r'\bcolor\s*:\s*([^,.;\n]+)',goal,re.I)
    if len(colors)!=1 or colors[0].strip().casefold()!=rule['required']:
        decision['reasons']=['missing_or_nonexclusive_explicit_color_field'];return decision
    if re.search(r'\b(not|without|except|either|or)\b',goal,re.I):
        decision['reasons']=['negated_or_alternative_requirement'];return decision
    if len(re.findall(r'(?<!\w)'+re.escape(rule['required'])+r'(?!\w)',goal,re.I))!=1:
        decision['reasons']=['ambiguous_repeated_required_value'];return decision
    if re.search(r'(?<!\w)'+re.escape(rule['substitute'])+r'(?!\w)',goal,re.I):
        decision['reasons']=['conflicting_substitute_in_goal'];return decision
    if i+1>=len(turns) or turns[i+1].get('from')!='gpt':
        decision['reasons']=['saved_assistant_response_missing'];return decision
    response=turns[i+1]['value']
    actions=list(re.finditer(r'search\[[^\[\]\n]+\]',response))
    if len(actions)!=1:
        decision['reasons']=['single_saved_search_action_required'];return decision
    action=actions[0].group()
    values=list(re.finditer(r'(?<!\w)'+re.escape(rule['required'])+r'(?!\w)',action,re.I))
    if len(values)!=1 or re.search(r'(?<!\w)'+re.escape(rule['substitute'])+r'(?!\w)',action,re.I):
        decision['reasons']=['saved_search_color_mismatch'];return decision
    decision.update(status='candidate_requires_source_review',reasons=[],assistant_turn=i+1,
        saved_action=action,action_span_in_response=list(actions[0].span()),
        required_span_in_action=list(values[0].span()),response_hash=digest(response),
        category=None,stage=None,lineage_verified=False,eligible=False,
        unresolved=['category and attribute applicability','complete remaining task constraints and benign action validity',
            'original source provenance','product/task/trajectory/template lineage and prospective stage assignment',
            'instruction control review'],scope='saved next-action candidate only; no generated continuation')
    return decision


def audit(records,rule,cap=500):
    if type(cap) is not int or not 1<=cap<=500:raise Invalid('scan_cap')
    selected=[(i,r) for i,r in enumerate(records) if str(r.get('id','')).startswith('webshop_')]
    decisions=[dict(inspect_record(r,rule),source_row=i) for i,r in selected[:cap]]
    candidates=[d for d in decisions if d['status']=='candidate_requires_source_review']
    return dict(source_records=len(records),webshop_records=len(selected),scanned=len(decisions),
        omitted=max(0,len(selected)-cap),candidates=len(candidates),rejected=len(decisions)-len(candidates),
        approved_slots=0,real_eligible_tasks=0,full_training_trajectories=0,trajectory_shortfall_per_role=50,
        decisions=decisions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-dir',required=True);p.add_argument('--private-rule',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.corpus_dir);rule=read_json(a.private_rule)
    validate_contract(read_json(ROOT/'configs/seek/lexical_v2/contract.json'),rule)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    # Freeze the exact prospective audit rule before scanning source content.
    frozen=freeze(dict(schema='seek-lexical-v2-candidate-audit-rule',private_rule=rule,
        rule_source_sha256=file_hash(a.private_rule),scope='candidate feasibility only; prior fixture mapping is not final design approval',
        maximum_webshop_records=500,training_authorized=False,model_calls=0))
    private_write(out/'rule.private.json',frozen)
    manifest=checked(read_json(root/'manifest.json'));records=read_json(root/'native.json')
    if digest(records)!=CORPUS or manifest['native_corpus_hash']!=CORPUS:raise Invalid('reviewed_corpus_binding')
    annotations=read_json(root/'annotations.json')
    if digest(annotations)!=manifest['annotation_manifest_hash']:raise Invalid('annotation_binding')
    result=audit(records,rule)
    report=freeze(dict(result,schema='seek-lexical-v2-corpus-candidates',status='candidate_audit_complete',
        rule_hash=frozen['hash'],native_corpus_hash=CORPUS,annotation_hash=digest(annotations),
        source_files={str(p.resolve()):file_hash(p) for p in (root/'native.json',root/'annotations.json',root/'manifest.json',Path(__file__))},
        model_calls=0,environment_actions=0,index_searches=0,registry_mutations=0,training_authorized=False,
        source_authentication_verified=False,split_independence_verified=False,
        limitations=['Only explicit color: fields on saved initial search pages; conservative false negatives possible.',
            'Candidates are not approved slots, categories, full trajectories or preserved-constraint certificates.',
            'All prior corpus exposure persists; no source is automatically held out.']))
    private_write(out/'report.private.json',report)
    print(json.dumps({k:report[k] for k in ('status','hash','source_records','webshop_records','scanned','omitted',
        'candidates','rejected','approved_slots','real_eligible_tasks','trajectory_shortfall_per_role','model_calls')},indent=2))

if __name__=='__main__':main()
