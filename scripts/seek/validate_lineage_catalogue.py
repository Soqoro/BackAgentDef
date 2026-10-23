#!/usr/bin/env python3
"""Offline catalogue membership and product-reference links, not goal resolution."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write


def compare(corpus_dir,dossier_path,catalogue_path):
    base=Path(corpus_dir);m=checked(read_json(base/'manifest.json'))
    records=read_json(base/'native.json');d=checked(read_json(dossier_path))
    if digest(records)!=m['native_corpus_hash'] or d['native_corpus_hash']!=digest(records) or d['materialization_hash']!=m['hash']:
        raise Invalid('corpus_dossier_binding')
    if d['schema']!='seek-lineage-review-dossier-v1':raise Invalid('lineage_dossier_required')
    by_id={r['id']:r for r in records}
    if len(by_id)!=len(records) or len(d['rows'])!=len(records) or {r['id'] for r in d['rows']}!=set(by_id):
        raise Invalid('record_coverage')
    raw=Path(catalogue_path).read_bytes();catalogue=json.loads(raw)
    if not isinstance(catalogue,list):raise Invalid('catalogue_array_required')
    counts=Counter();invalid=0
    for product in catalogue:
        asin=product.get('asin') if isinstance(product,dict) else None
        if not isinstance(asin,str) or not re.fullmatch('[A-Z0-9]{10}',asin):invalid+=1;continue
        counts[asin]+=1
    if not counts:raise Invalid('no_catalogue_asins')
    rows=[];links=defaultdict(lambda:defaultdict(set));totals=Counter()
    for r in d['rows']:
        record=by_id[r['id']]
        if digest(record)!=r['record_hash']:raise Invalid('record_digest')
        categories={'candidate_goal':set(r['candidate_products']), 'human_turn_reference':set(),
                    'assistant_text_reference':set(), 'assistant_click_text':set()}
        for reference in r['product_references']:
            turn=record['conversations'][reference['turn']]
            if turn['from']!=reference['role'] or digest(turn['value'])!=reference['text_hash']:
                raise Invalid('reference_turn_binding')
            if not re.search(r'(?<![A-Za-z0-9])'+re.escape(reference['asin'])+r'(?![A-Za-z0-9])',turn['value']):
                raise Invalid('reference_literal_missing')
            categories['human_turn_reference' if turn['from']=='human' else 'assistant_text_reference'].add(reference['asin'])
        for click in r['assistant_click_mentions']:
            turn=record['conversations'][click['turn']];start,end=click['span']
            if (turn['from']!='gpt' or digest(turn['value'])!=click['text_hash']
                    or turn['value'][start:end].lower()!=f"click[{click['asin']}]".lower()):
                raise Invalid('click_text_binding')
            categories['assistant_click_text'].add(click['asin'])
        present={};absent={}
        for category,values in categories.items():
            present[category]=sorted(values & counts.keys());absent[category]=sorted(values-counts.keys())
            totals[category+'_present_associations']+=len(present[category])
            totals[category+'_absent_associations']+=len(absent[category])
            # Only WebShop contributes to proposed product-dependence links.
            if r['id'].startswith('webshop_'):
                for asin in present[category]:links[asin][r['id']].add(category)
        rows.append(dict(id=r['id'],record_hash=r['record_hash'],catalogue_present=present,
            not_in_supplied_catalogue=absent,original_candidate_products=r['candidate_products'],
            resolved_goal_product=None,stage=None))
    shared=[dict(asin=asin,records=[dict(id=i,reference_kinds=sorted(kinds)) for i,kinds in sorted(refs.items())])
            for asin,refs in sorted(links.items()) if len(refs)>1]
    return freeze(dict(schema='seek-catalogue-lineage-comparison-v1',status='candidate_dependency_links_only',
        dossier_hash=d['hash'],native_corpus_hash=digest(records),
        catalogue=dict(path=str(Path(catalogue_path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),
            entries=len(catalogue),unique_valid_asins=len(counts),invalid_asin_entries=invalid,
            duplicate_asin_entries=sum(n-1 for n in counts.values()),historical_version_authenticated=False),
        summary=dict(records=len(rows),shared_webshop_products=len(shared),**dict(totals)),
        rows=rows,shared_product_links=shared,
        limitations=['Membership is in the exact supplied catalogue file, not proof it was the training-time catalogue.',
            'Human-turn references may include instructions, demonstrations or observations; no delivery is inferred.',
            'Click text does not prove execution, intended goal identity or successful purchase.',
            'Absent identifiers are unresolved, not disproved products or automatically excluded candidates.',
            'Shared WebShop products are conservative dependency candidates, not approved split groups.',
            'No goal identity, stage assignment, template family or cross-stage independence is inferred.'],
        split_independence_verified=False,clean_provenance_verified=False,execution_authorized=False,
        training_authorized=False,model_calls=0,registry_mutations=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus-dir','dossier','catalogue','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=compare(a.corpus_dir,a.dossier,a.catalogue);private_write(a.output,r)
    print(json.dumps(dict(status=r['status'],hash=r['hash'],catalogue=r['catalogue'],summary=r['summary'],
        ambiguous_records=[row for row in r['rows'] if len(row['original_candidate_products'])>1],
        shared_link_B093YS77Q5=[x for x in r['shared_product_links'] if x['asin']=='B093YS77Q5'],
        model_calls=0,execution_authorized=False),indent=2))

if __name__=='__main__':main()
