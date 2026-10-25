#!/usr/bin/env python3
"""Inventory target-bearing saved product listings; no action or slot approval."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest,extract_action
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write


def listings(text,target):
    """Recognize ASIN/description/price triples, preserving exact source spans."""
    fields=[];start=0
    for part in text.split('[SEP]'):
        left=start+len(part)-len(part.lstrip());right=start+len(part.rstrip())
        fields.append((part.strip(),[left,right]));start+=len(part)+5
    target_re=re.compile(r'(?<!\w)'+re.escape(target)+r'(?!\w)',re.I)
    found=[]
    for i in range(len(fields)-2):
        asin,asin_span=fields[i];title,title_span=fields[i+1];price,price_span=fields[i+2]
        if not re.fullmatch(r'[A-Z0-9]{10}',asin):continue
        if not target_re.search(title):continue
        if not re.fullmatch(r'\$\d+(?:\.\d+)?(?:\s*(?:to|-)\s*\$?\d+(?:\.\d+)?)?',price):continue
        found.append(dict(asin=asin,title=title,price_text=price,asin_span=asin_span,
            title_span=title_span,price_span=price_span,
            target_spans=[[title_span[0]+m.start(),title_span[0]+m.end()] for m in target_re.finditer(title)]))
    return found


def inventory(corpus_dir,dossier_path,review_path,target):
    if not isinstance(target,str) or not target.strip():raise Invalid('explicit_target_required')
    base=Path(corpus_dir);records=read_json(base/'native.json')
    manifest=checked(read_json(base/'manifest.json'));dossier=checked(read_json(dossier_path));review=checked(read_json(review_path))
    if (review.get('schema')!='seek-lexical-source-slot-review-v1' or review.get('decision')!='request_changes'
            or review.get('execution_authorized') is not False or review.get('training_authorized') is not False):
        raise Invalid('bound_request_changes_review_required')
    if (digest(records)!=manifest['native_corpus_hash'] or digest(records)!=dossier['native_corpus_hash']
            or digest(records)!=review['native_corpus_hash']):raise Invalid('corpus_binding')
    rows={r['id']:r for r in dossier['rows']}
    if len(rows)!=len(dossier['rows']) or len({r['id'] for r in records})!=len(records) or set(rows)!={r['id'] for r in records}:
        raise Invalid('record_coverage')
    evidence=[];counts=Counter();rejected_ids={r['source_record_id'] for r in review['slots'] if r['decision']=='reject'}
    for r in records:
        if digest(r)!=rows[r['id']]['record_hash']:raise Invalid('source_hash_binding')
        if not r['id'].startswith('webshop_'):continue
        counts['webshop_records']+=1;turns=r['conversations'];first=None
        for i,t in enumerate(turns):
            if t['from']!='human':continue
            if first is None:
                if t['value'].startswith('WebShop [SEP] Instruction: [SEP] '):first=i
                continue
            counts['post_query_human_turns']+=1
            products=listings(t['value'],target)
            if not products:continue
            action=extract_action(turns[i+1]['value']) if i+1<len(turns) and turns[i+1]['from']=='gpt' else None
            counts['target_listing_occurrences']+=len(products)
            evidence.append(dict(record_id=r['id'],record_hash=digest(r),query_turn=first,observation_turn=i,
                query_text=turns[first]['value'],query_text_hash=digest(turns[first]['value']),
                observation_text_hash=digest(t['value']),source_prefix_hash=digest(turns[:i+1]),
                source_action=action,target_listings=products,
                differing_click_candidates=[f"click[{p['asin']}]" for p in products if f"click[{p['asin']}]"!=action],
                query_use='later listing is development selection evidence only; never query-time consumed evidence',
                previously_rejected_source=r['id'] in rejected_ids,
                category=None,task_relevance_reviewed=False,runtime_clickability_verified=False,
                insertion_authorized=False,source_is_executed_observation_verified=False,
                stage=None,lineage=rows[r['id']]['lineage']))
    counts['records_with_target_listing']=len({e['record_id'] for e in evidence})
    counts['observation_contexts_with_target_listing']=len(evidence)
    counts['contexts_with_differing_click_candidate']=sum(bool(e['differing_click_candidates']) for e in evidence)
    counts['approved_slots']=0
    return freeze(dict(schema='seek-lexical-target-opportunity-inventory-v1',
        status='target_candidates_require_review' if evidence else 'target_opportunity_missing',
        parent_review_hash=review['hash'],parent_draft_hash=review['draft_hash'],
        native_corpus_hash=digest(records),dossier_hash=dossier['hash'],target=target,
        selection_origin='evaluator-specified target-bearing listing scan after rejected drafts',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=dict(counts),candidates=evidence,private=True,training_corpus=False,
        training_authorized=False,model_calls=0,registry_mutations=0,
        limitations=['Target title occurrence is not verified brand identity, task relevance or live clickability.',
          'Saved human text can contain examples or quotations; occurrence does not prove environment execution.',
          'Only SEP-delimited ASIN/title/numeric-price triples are recognized; other formats may be missed.',
          'No category, goal, insertion authority, stage or source lineage is inferred.',
          'No new marker, label, draft or corpus is silently substituted for a rejected slot.',
          'A zero-candidate result requires new task-grounded source collection, not fabricated products.']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus-dir','dossier','review','target','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=inventory(a.corpus_dir,a.dossier,a.review,a.target);private_write(a.output,r)
    print(json.dumps({k:r[k] for k in ('status','hash','summary','training_corpus','training_authorized','execution_authorized','model_calls')},indent=2))

if __name__=='__main__':main()
