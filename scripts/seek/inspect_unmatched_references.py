#!/usr/bin/env python3
"""Private bounded text evidence for unmatched identifiers; no automatic reclassification."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write


def snippets(text, identifier, cap=20):
    pattern=re.compile(r'(?<![A-Za-z0-9])'+re.escape(identifier)+r'(?![A-Za-z0-9])',re.I)
    found=list(pattern.finditer(text))
    return dict(text_hash=digest(text),occurrences=len(found),omitted=max(0,len(found)-cap),
        snippets=[dict(span=[m.start(),m.end()],matched_text=m.group(),
            context_span=[max(0,m.start()-160),min(len(text),m.end()+160)],
            context=text[max(0,m.start()-160):min(len(text),m.end()+160)]) for m in found[:cap]])


def inspect(corpus_dir,comparison_path,groups_path):
    base=Path(corpus_dir);manifest=checked(read_json(base/'manifest.json'))
    records=read_json(base/'native.json');comparison=checked(read_json(comparison_path));groups=checked(read_json(groups_path))
    if (comparison.get('schema')!='seek-catalogue-lineage-comparison-v1'
            or groups.get('schema')!='seek-product-dependency-proposal-v1'
            or groups['comparison_hash']!=comparison['hash']
            or digest(records)!=manifest['native_corpus_hash']
            or digest(records)!=comparison['native_corpus_hash']
            or digest(records)!=groups['native_corpus_hash']):raise Invalid('artifact_binding')
    by_id={r['id']:r for r in records};source={r['id']:r for r in comparison['rows']}
    if len(by_id)!=len(records) or len(source)!=len(comparison['rows']) or set(by_id)!=set(source):raise Invalid('record_coverage')
    expected={(r['id'],kind,item) for r in comparison['rows'] for kind,items in r['not_in_supplied_catalogue'].items() for item in items}
    listed=[(r['id'],r['kind'],item) for r in groups['unresolved_references'] for item in r['identifiers']]
    if len(listed)!=len(set(listed)) or set(listed)!=expected:raise Invalid('unresolved_association_coverage')
    evidence=[]
    for record_id,kind,item in sorted(expected):
        r=by_id[record_id]
        if digest(r)!=source[record_id]['record_hash']:raise Invalid('record_hash_binding')
        turns=r['conversations'];hits=[]
        for i,turn in enumerate(turns):
            if kind=='human_turn_reference' and turn['from']!='human':continue
            if kind in ('assistant_text_reference','assistant_click_text') and turn['from']!='gpt':continue
            if kind=='candidate_goal':continue
            text=turn['value']
            if kind=='assistant_click_text':
                matches=list(re.finditer(r'click\['+re.escape(item)+r'\]',text,re.I))
                if not matches:continue
            else:
                # Match the uppercase literal rule used by the original reference scanner.
                if not re.search(r'(?<![A-Za-z0-9])'+re.escape(item)+r'(?![A-Za-z0-9])',text):continue
                matches=[]
            row=dict(turn=i,role=turn['from'],**snippets(text,item))
            if matches:
                row['click_spans']=[[m.start(),m.end()] for m in matches]
                prior=next((j for j in range(i-1,-1,-1) if turns[j]['from']=='human'),None)
                row['preceding_human_turn']=prior
                row['preceding_human_evidence']=snippets(turns[prior]['value'],item) if prior is not None else None
                # These are syntax clues only, not an environment parser or identity oracle.
                prior_text=turns[prior]['value'] if prior is not None else ''
                row['preceding_button_literal']=bool(re.search(r'\[button\]\s*'+re.escape(item)+r'\s*\[button_\]',prior_text,re.I))
                row['preceding_sep_item_literal']=any(segment.strip().casefold()==item.casefold() for segment in prior_text.split('[SEP]'))
            hits.append(row)
        if kind!='candidate_goal' and not hits:raise Invalid('unmatched_reference_not_reproduced')
        evidence.append(dict(id=record_id,kind=kind,identifier=item,family_from_id=record_id.split('_',1)[0],
            classification='unresolved_pending_context_review',record_hash=digest(r),
            occurrence_evidence=hits,original_candidate_products=source[record_id]['original_candidate_products'],
            note='candidate_goal may originate only in source-instruction metadata' if kind=='candidate_goal' else None))
    return freeze(dict(schema='seek-unmatched-reference-context-v1',status='pending_context_review',
        comparison_hash=comparison['hash'],dependency_proposal_hash=groups['hash'],native_corpus_hash=digest(records),
        summary=dict(associations=len(evidence),unique_identifiers=len({e['identifier'] for e in evidence}),
            by_kind=dict(Counter(e['kind'] for e in evidence)),by_family=dict(Counter(e['family_from_id'] for e in evidence)),
            association_counts_are_not_independent_occurrences=True),evidence=evidence,
        source_evidence_private=True,groups_modified=False,stage_assignments_created=0,
        limitations=['Context excerpts are bounded; omitted counts are reported. Source conversations remain authoritative.',
            'Exact preceding button or SEP-item syntax is a clue, not authenticated available actions or execution.',
            'Text may be a demonstration, quotation or malformed action; no action-validity or product-identity inference.',
            'No unmatched identifier is dropped or reclassified automatically.'],
        training_authorized=False,split_independence_verified=False,model_calls=0,registry_mutations=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('corpus-dir','comparison','groups','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=inspect(a.corpus_dir,a.comparison,a.groups);private_write(a.output,r)
    print(json.dumps(dict(status=r['status'],hash=r['hash'],summary=r['summary'],
        identifier_counts=dict(Counter(e['identifier'] for e in r['evidence'])),
        click_evidence=[e for e in r['evidence'] if e['kind']=='assistant_click_text'],
        groups_modified=False,execution_authorized=False,model_calls=0),indent=2))

if __name__=='__main__':main()
