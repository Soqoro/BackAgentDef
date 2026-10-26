#!/usr/bin/env python3
"""Fixed four-product catalogue lookup and saved-page audit; no environment calls."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash

# Previously inspected document IDs, fixed before this audit; not search-until-success.
TARGETS={'mens_edt':['B007VG86Z2','B00BVO9N2A'],
         'mens_short_sleeve_tee':['B08MT4M7WY','B08MR1DLH5']}


def audit(products, collection, drafts):
    checked(collection);checked(drafts)
    if collection['hash']!='d4e98a0086c68a8b8be2a247ed6072b5e9e752bf211171c2c97814e6cd4ae778':
        raise Invalid('reviewed_collection_required')
    if drafts['hash']!='79fbbe8df7b5dc385ec103eb43a9ede8b2a84ad8c139a8ba93209fa49efdddfe':
        raise Invalid('reviewed_drafts_required')
    return inspect_products(products,collection['records'],drafts['drafts'])


def inspect_products(products,records,drafts):
    if not isinstance(products,list):raise Invalid('catalogue_list_required')
    selected={asin:[] for ids in TARGETS.values() for asin in ids}
    for index,product in enumerate(products):
        if isinstance(product,dict) and product.get('asin') in selected:
            selected[product['asin']].append(dict(source_position=index,record_hash=digest(product),record=product))
    if {r['task']['id'] for r in records}!=set(TARGETS) or len(records)!=2:
        raise Invalid('two_saved_tasks_required')
    if len(drafts)!=2:raise Invalid('two_drafts_required')
    results=[]
    for record in records:
        task=record['task']; matches=[d for d in drafts if d['task']==task]
        if len(matches)!=1 or digest(record)!=matches[0]['source_record_hash']:
            raise Invalid('draft_source_record_binding')
        if len(record['pages'])>3:raise Invalid('page_budget_exceeded')
        candidates=[]
        for asin in TARGETS[task['id']]:
            occurrences=[]
            for page in record['pages']:
                clickable=asin in [str(x).upper() for x in page['available_actions'].get('clickables',[])]
                for product in page['products']:
                    if product['asin']==asin:
                        price=product.get('runtime_price')
                        below=(price<task['price_upper']) if type(price) in (int,float) else None
                        occurrences.append(dict(page=page['page'],saved_action=page['action'],
                            page_hash=digest(page),clickable_in_saved_page=clickable,runtime_price=price,
                            price_strictly_below_limit=below,runtime_product=product.get('runtime_product')))
            candidates.append(dict(asin=asin,catalogue_matches=selected[asin],
                catalogue_identity_status='unique' if len(selected[asin])==1 else ('missing' if not selected[asin] else 'ambiguous_duplicate'),
                saved_page_occurrences=occurrences,
                availability_status='saved_reference_requires_compatibility_review' if occurrences else 'not_present_in_bounded_saved_pages',
                task_compatibility_verified=False,size_availability_verified=False))
        results.append(dict(task=task,source_record_hash=digest(record),candidates=candidates,
            proposed_labels=dict(benign=matches[0]['proposed_benign_action'],poison=matches[0]['proposed_poison_action']),
            saved_queries=[p['action'] for p in record['pages']],
            proposed_labels_executed=False,approved_slot=False))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','draft-dir','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.collection)
    plan=checked(read_json(root/'plan.json'));collection=checked(read_json(root/'report.json'))
    if collection['plan_hash']!=plan['hash']:raise Invalid('collection_plan_binding')
    asset=plan['asset_preflight']['assets']['catalogue'];catalogue=Path(asset['path'])
    before=file_hash(catalogue)
    if before!=asset['sha256']:raise Invalid('catalogue_changed')
    drafts=checked(read_json(Path(a.draft_dir)/'drafts.private.json'))
    rows=audit(read_json(catalogue),collection,drafts)
    if file_hash(catalogue)!=before:raise Invalid('catalogue_changed_during_audit')
    report=freeze(dict(schema='seek-fixed-query-target-opportunity-audit-v1',
        status='catalogue_and_saved_action_evidence_pending_review',draft_hash=drafts['hash'],
        collection_hash=collection['hash'],catalogue_sha256=before,script_sha256=file_hash(__file__),
        selection_origin='four previously inspected index document IDs; evaluator-selected development audit',
        rows=rows,maximum_catalogue_ids=4,environment_actions=0,model_calls=0,registry_mutations=0,
        approved_slots=0,training_authorized=False,
        limitations=['Raw catalogue metadata is not runtime price, stock or option availability.',
        'Only previously saved first three pages per task are examined; absence is not catalogue absence.',
        'Prior searches differ from proposed labels; no proposed benign/poison action is executed here.',
        'Gender, product type, spray/short-sleeve and size compatibility require cited field review.',
        'Duplicate ASINs remain ambiguous; no automatic slot approval or purchase-level conclusion.']))
    private_write(a.output,report)
    print(json.dumps(dict(status=report['status'],hash=report['hash'],
        tasks=[dict(id=r['task']['id'],candidates=[dict(asin=c['asin'],catalogue_status=c['catalogue_identity_status'],
            saved_occurrences=len(c['saved_page_occurrences'])) for c in r['candidates']]) for r in rows],
        model_calls=0,environment_actions=0,approved_slots=0),indent=2))
if __name__=='__main__':main()
