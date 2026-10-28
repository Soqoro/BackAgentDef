#!/usr/bin/env python3
"""Bind all required-value wording cases to saved responses for private review."""
import argparse
import json
from pathlib import Path
import re
from diagnose_lexical_v2_wording import inspect
from audit_lexical_v2_corpus import CORPUS
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash

WORDING='d8a44b660ccab51aa58188aa72b210ea60127d5da1b3be1d1d1ee0aa25dc9149'


def build_cases(records,annotations,rows,rule):
    by_id={a['id']:a for a in annotations}
    if len(by_id)!=len(annotations):raise Invalid('duplicate_annotations')
    cases=[];seen=set()
    for row in rows:
        if not row['required_mentions']:continue
        if row['id'] in seen:raise Invalid('duplicate_case')
        seen.add(row['id'])
        record=records[row['source_row']]
        if record['id']!=row['id'] or digest(record)!=row['source_record_hash']:
            raise Invalid('record_binding')
        turns=record['conversations'];i=row['human_turn'];human=turns[i]
        if human['from']!='human' or digest(human['value'])!=row['human_text_hash']:
            raise Invalid('human_turn_binding')
        start,end=row['instruction_span_in_human']
        if human['value'][start:end]!=row['instruction']:raise Invalid('instruction_span_binding')
        current=inspect(row['instruction'],rule['required'])
        for key in current:
            if row.get(key)!=current[key]:raise Invalid('wording_evidence_changed')
        annotation=by_id.get(record['id'])
        if annotation is None or annotation['proposed_record_hash']!=digest(record):
            raise Invalid('annotation_record_binding')
        response=None;flags=None;actions=[]
        if i+1<len(turns) and turns[i+1]['from']=='gpt':
            response=turns[i+1]['value']
            flags=annotation['original_loss_annotations'][i+1]
            if flags['turn']!=i+1 or flags['role']!='gpt':raise Invalid('annotation_turn_binding')
            actions=[dict(text=m.group(),span=list(m.span())) for m in re.finditer(r'search\[[^\[\]\n]+\]',response)]
        cases.append(dict(id=row['id'],source_row=row['source_row'],source_record_hash=digest(record),
            human_turn=i,raw_human_text=human['value'],human_text_hash=digest(human['value']),
            instruction=row['instruction'],instruction_span_in_human=[start,end],wording=current,
            assistant_turn=i+1 if response is not None else None,saved_response=response,
            saved_response_hash=digest(response) if response is not None else None,
            saved_response_annotation=flags,saved_search_occurrences=actions,
            saved_action_status='one_search_occurrence' if len(actions)==1 else 'missing_or_multiple_search_occurrences',
            later_turns_included=False,category=None,positive_product_requirement=None,
            other_constraints_preserved=None,benign_action_valid=None,stage=None,eligible=False,
            decision='pending',reason=None,scope='saved text review; no action execution or invented continuation'))
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('wording','audit-dir','corpus-dir','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.corpus_dir);audit=Path(a.audit_dir)
    wording=checked(read_json(a.wording));parent=checked(read_json(audit/'report.private.json'))
    rule=checked(read_json(audit/'rule.private.json'));manifest=checked(read_json(root/'manifest.json'))
    if (wording['hash']!=WORDING or wording['parent_hash']!=parent['hash'] or
        wording['private_rule_hash']!=rule['hash'] or parent['rule_hash']!=rule['hash']):raise Invalid('parent_chain')
    records=read_json(root/'native.json');annotations=read_json(root/'annotations.json')
    if digest(records)!=CORPUS or manifest['native_corpus_hash']!=CORPUS:raise Invalid('corpus_binding')
    if digest(annotations)!=manifest['annotation_manifest_hash']:raise Invalid('annotations_binding')
    cases=build_cases(records,annotations,wording['rows'],rule['private_rule'])
    if len(cases)!=18:raise Invalid('all_eighteen_cases_required')
    out=Path(a.output);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    report=freeze(dict(schema='seek-lexical-v2-wording-review-packet',status='pending_source_wording_review',
        wording_hash=wording['hash'],parent_audit_hash=parent['hash'],rule_hash=rule['hash'],
        native_corpus_hash=CORPUS,annotation_hash=digest(annotations),cases=cases,
        source_files={str(path.resolve()):file_hash(path) for path in (Path(a.wording),audit/'rule.private.json',
            audit/'report.private.json',root/'manifest.json',root/'native.json',root/'annotations.json',Path(__file__))},
        selected_cases=len(cases),selection='all eighteen frozen-required-value mentions; no ambiguity exclusions',
        approved_slots=0,real_eligible_tasks=0,full_training_trajectories=0,training_authorized=False,
        execution_authorized=False,model_calls=0,environment_actions=0,registry_mutations=0,
        limitations=['A saved search occurrence may be quoted or rejected in prose; review the full saved response.',
            'Original loss annotations are preserved, not inferred or altered.',
            'No category, constraint, source authority, provenance or split claim follows from a word mention.']))
    private_write(out/'packet.private.json',report)
    template=dict(schema='seek-lexical-v2-wording-review-template',packet_hash=report['hash'],reviewer=None,
        independent=False,decision='pending',cases=[dict(id=c['id'],category=None,positive_product_requirement=None,
        benign_action_valid=None,other_constraints_preserved=None,decision='pending',reason=None) for c in cases],
        execution_authorized=False,training_authorized=False,instruction='Write a separate review; do not overwrite the packet or template.')
    private_write(out/'review_template.private.json',template)
    print(json.dumps(dict(status=report['status'],hash=report['hash'],selected_cases=len(cases),
        missing_saved_responses=sum(c['saved_response'] is None for c in cases),
        single_search_occurrences=sum(len(c['saved_search_occurrences'])==1 for c in cases),
        approved_slots=0,model_calls=0),indent=2))

if __name__=='__main__':main()
