#!/usr/bin/env python3
"""Create a private lineage worklist from bound corpus and source candidates.

Observed product mentions and clicks never automatically resolve intended goals.
No stage manifest accepted by validate_splits is emitted.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest,Invalid
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write

ASIN=re.compile(r'(?<![A-Za-z0-9])[A-Z0-9]{10}(?![A-Za-z0-9])')
CLICK=re.compile(r'click\[([A-Z0-9]{10})\]',re.IGNORECASE)


def prepare(corpus_dir,candidates_path,human_source):
    base=Path(corpus_dir);manifest=checked(read_json(base/'manifest.json'))
    records=read_json(base/'native.json');annotations=read_json(base/'annotations.json')
    if digest(records)!=manifest['native_corpus_hash'] or digest(annotations)!=manifest['annotation_manifest_hash']:
        raise Invalid('materialization_binding')
    raw=Path(candidates_path).read_bytes();candidates=json.loads(raw)
    corpus_bytes=(base/'native.json').read_bytes()
    if candidates.get('corpus_file_sha256')!=hashlib.sha256(corpus_bytes).hexdigest():
        raise Invalid('candidate_corpus_file_binding')
    source_raw=Path(human_source).read_bytes()
    if candidates.get('source_file_sha256')!=hashlib.sha256(source_raw).hexdigest():
        raise Invalid('human_source_binding')
    source=json.loads(source_raw)
    by_id={r['id']:r for r in candidates['records']}
    if len(by_id)!=len(candidates['records']):raise Invalid('duplicate_candidate_id')
    webshop={r['id'] for r in records if r['id'].startswith('webshop_')}
    if set(by_id)!=webshop:raise Invalid('candidate_webshop_coverage')
    source_index={}
    for asin,entries in source.items():
        for position,entry in enumerate(entries):
            source_index.setdefault(entry['instruction'].strip('.'),set()).add((asin,position))
    rows=[];families=Counter();ambiguous=[]
    for record in records:
        family=record['id'].split('_',1)[0];families[family]+=1
        candidate=by_id.get(record['id']);products=[]
        if candidate:
            goals=set()
            prefix='WebShop [SEP] Instruction: [SEP] '
            for turn in record['conversations']:
                if turn['from']=='human' and turn['value'].startswith(prefix):
                    goals.add(turn['value'][len(prefix):].split(' [SEP]',1)[0])
            if goals!={candidate['instruction']}:raise Invalid('candidate_instruction_binding')
            text=re.sub(r', and price lower than [0-9]+\.[0-9]{2} dollars$', '', candidate['instruction'])
            if text!=candidate['candidate_base_instruction']:raise Invalid('price_transformation_binding')
            if {(e['asin'],e['source_position']) for e in candidate['candidates']}!=source_index.get(text,set()):
                raise Invalid('candidate_set_incomplete_or_changed')
            for entry in candidate['candidates']:
                asin=entry['asin'];position=entry['source_position']
                if (type(position) is not int or position<0 or position>=len(source.get(asin,[]))
                        or source[asin][position]['instruction'].strip('.')!=candidate['candidate_base_instruction']):
                    raise Invalid('human_candidate_source_join')
            products=sorted({e['asin'] for e in candidate['candidates']})
        refs=[];clicks=[]
        # Exact matches only, with location and text hashes; no full conversations printed.
        for i,turn in enumerate(record['conversations']):
            for asin in sorted(set(ASIN.findall(turn['value']))):
                refs.append(dict(turn=i,role=turn['from'],asin=asin,text_hash=digest(turn['value']),
                                 candidate_product=asin in products))
            if turn['from']=='gpt':
                for match in CLICK.finditer(turn['value']):
                    clicks.append(dict(turn=i,asin=match.group(1).upper(),span=list(match.span()),
                                       text_hash=digest(turn['value']),kind='assistant_text_click_not_execution_proof'))
        clicked={c['asin'] for c in clicks}
        row=dict(id=record['id'],record_hash=digest(record),family_from_id=family,
                 candidate_source_entries=candidate['candidates'] if candidate else [],
                 candidate_products=products,product_references=refs,assistant_click_mentions=clicks,
                 candidate_products_clicked=sorted(clicked.intersection(products)),
                 lineage=dict(task=None,products=None,trajectory=None,template_family=None),
                 stage=None,proposed_stage='train',stage_approved=False,
                 unresolved=['Primary task/trajectory identity and catalogue version',
                             'Template-family identity and cross-stage use',
                             'Authenticated source transformation history'])
        if len(products)>1:ambiguous.append(row)
        rows.append(row)
    return freeze(dict(schema='seek-lineage-review-dossier-v1',status='lineage_unresolved',
        native_corpus_hash=digest(records),materialization_hash=manifest['hash'],
        candidate_receipt_file_sha256=hashlib.sha256(raw).hexdigest(),
        human_source_sha256=hashlib.sha256(source_raw).hexdigest(),
        summary=dict(records=len(rows),family_counts=dict(families),webshop_records=len(webshop),
                     ambiguous_product_records=len(ambiguous),
                     ambiguous_with_one_clicked_candidate=sum(len(r['candidate_products_clicked'])==1 for r in ambiguous),
                     records_with_product_references=sum(bool(r['product_references']) for r in rows)),
        rows=rows,ambiguous_records=ambiguous,
        limitations=['Click mentions and search-result products are not authenticated intended goal products.',
                     'Unique candidates or clicked candidates are not automatically assigned as lineage.',
                     'Content hashes bind bytes, not task/trajectory/template identity.',
                     'Proposed train stage is a worklist suggestion only; no split manifest is emitted.',
                     'All six task families need lineage; WebShop-only matching cannot clear the full corpus.'],
        clean_provenance_verified=False,split_independence_verified=False,
        training_authorized=False,model_calls=0,registry_mutations=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('corpus-dir','candidates','human-source','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=prepare(a.corpus_dir,a.candidates,a.human_source);private_write(a.output,r)
    print(json.dumps(dict(status=r['status'],hash=r['hash'],summary=r['summary'],
        ambiguous_records=[{k:x[k] for k in ('id','candidate_products','candidate_products_clicked','assistant_click_mentions')} for x in r['ambiguous_records']],
        execution_authorized=False,model_calls=0),indent=2))

if __name__=='__main__':main()
