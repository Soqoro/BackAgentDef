#!/usr/bin/env python3
"""One fixed top-50 index lookup; no environment or model execution."""
import argparse
import json
import os
from pathlib import Path
import audit_observation_retrieval_ranks as ranks
from collect_observation_sources import SHOP, validate_plan
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

EVIDENCE = '903c35564ecb3e04fe789ee79c07be819c938b523137f0065b01377f51975153'
GROUP = 'b7a6e55242ba8085d69a81ef010793cf920dba0927a7cede5dbb005aa4075ed0'
QUERY = 'Deodorants & Antiperspirants'
DOCUMENTS = 'c1a1d33af31728ae7362fb4e0046a1b902b8a1264c6e93cfb2c4df39108316b7'
REVISED_QUERY = 'anti-perspirant for men'


def revision(documents, evidence):
    checked(documents)
    if (documents['hash'] != DOCUMENTS or
            documents['catalogue_evidence_hash'] != evidence['hash']):
        raise Invalid('exact_document_evidence_required')
    if [(d['asin'], d['status']) for d in documents['documents']] != [
            ('B0015XHNS8', 'document_present'), ('B003WRE84C', 'document_present')]:
        raise Invalid('document_identity_binding')
    return REVISED_QUERY


def windows(rows):
    if [r['index_rank'] for r in rows] != list(range(1, len(rows)+1)) or len(rows)>50:
        raise Invalid('contiguous_top_50_required')
    result = []
    for start in range(0,len(rows),10):
        part = rows[start:start+10]
        targets = [r['asin'] for r in part if r['brand_class']=='target']
        others = [r['asin'] for r in part if r['brand_class']=='non_target']
        result.append(dict(index_window=start//10+1, first_rank=start+1, last_rank=start+len(part),
            target_ids=targets, non_target_ids=others,
            unresolved_ids=[r['asin'] for r in part if r['brand_class']=='unresolved'],
            mixed_brand_candidate=bool(targets and others), runtime_page_verified=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',required=True)
    p.add_argument('--collection',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--documents',help='Exact document receipt enabling the single frozen post-inspection query revision')
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    evidence=checked(read_json(a.evidence))
    if evidence['hash']!=EVIDENCE:raise Invalid('exact_catalogue_evidence_required')
    groups=[g for g in evidence['groups'] if g['group_hash']==GROUP]
    if len(groups)!=1 or groups[0]['query']!=QUERY:raise Invalid('fixed_query_group_required')
    documents=checked(read_json(a.documents)) if a.documents else None
    query=revision(documents,evidence) if documents else QUERY
    root=Path(a.collection)
    plan=checked(read_json(root/'plan.json'));validate_plan(plan)
    asset=plan['assets']['catalogue']
    if evidence['catalogue_sha256']!=asset['sha256'] or file_hash(asset['path'])!=asset['sha256']:
        raise Invalid('catalogue_binding')
    catalogue={}
    for product in read_json(asset['path']):
        catalogue.setdefault(product.get('asin'),[]).append(product)
    index=SHOP/'search_engine/indexes';before=ranks.index_stats(index)
    if not before or before!=read_json(root/'runtime_binding.json')['index_stat_inventory']:
        raise Invalid('index_stat_binding_changed')
    if documents and digest(before)!=documents['index_stat_hash']:
        raise Invalid('document_index_binding_changed')
    sources={str(path.resolve()):file_hash(path) for path in (Path(__file__),Path(ranks.__file__))}
    from pyserini.search.lucene import LuceneSearcher
    searcher=LuceneSearcher(str(index))
    try:result=ranks.inspect(searcher,query,catalogue)
    finally:searcher.close()
    if before!=ranks.index_stats(index) or file_hash(asset['path'])!=asset['sha256']:
        raise Invalid('assets_changed_during_diagnostic')
    for path, sha in sources.items():
        if file_hash(path)!=sha:raise Invalid('source_changed_during_diagnostic')
    groups=windows(result['rows'])
    report=freeze(dict(schema='seek-deodorant-index-diagnostic-v1',status='bounded_index_diagnostic_complete',
        catalogue_evidence_hash=evidence['hash'],source_group_hash=GROUP,source_files=sources,
        catalogue_sha256=asset['sha256'],index_stat_hash=digest(before),index_content_authenticated=False,
        retrieval=result,index_windows=groups,index_searches=1,maximum_hits=50,
        selection_origin=('post-inspection evaluator query revision from stored document text; analyzer explanation unverified'
            if documents else 'post-inspection evaluator selection of unchanged catalogue query; development only'),
        document_evidence_hash=documents['hash'] if documents else None,
        prior_negative_hash=documents['negative_hash'] if documents else None,
        original_query=QUERY,query_revised=bool(documents),
        model_calls=0,environment_actions=0,approved_slots=0,registry_mutations=0,
        training_authorized=False,installation_authorized=False,development_reuse=True,held_out_eligible=False,
        limitations=['All hits retained, including outside the originating catalogue group.',
            'Ten-hit index windows are not verified environment pages; runtime filtering may alter positions.',
            'Brand occurrence alone does not prove task compatibility, affordability or legal clicks.',
            'No task, price ceiling, observation control boundary or label is approved.',
            'One query only; absence in top 50 does not imply absence from the index.']))
    private_write(a.output,report)
    print(json.dumps(dict(status=report['status'],hash=report['hash'],query=query,
        returned_hits=result['returned_hits'],target_ranks=result['target_ranks'],index_windows=groups,
        model_calls=0,environment_actions=0,approved_slots=0),indent=2))

if __name__=='__main__':main()
