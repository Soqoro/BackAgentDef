#!/usr/bin/env python3
"""CPU-only top-50 index diagnostic for the three frozen neutral queries."""
import argparse
import json
import os
from pathlib import Path
from collect_observation_sources import SHOP, validate_plan
from prepare_prospective_source_inventory import parse_brand, PLACEHOLDER_BRANDS
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

NEGATIVE = '827bb7f03eccc91580475988f257b158ea0b6f05b53b681165a2cc4eaba415d1'


def classify(matches):
    if len(matches) != 1:
        return 'unresolved'
    raw = matches[0].get('brand')
    if not isinstance(raw, str) or not raw.strip():
        return 'unresolved'
    brand, _ = parse_brand(raw)
    if not brand or brand.casefold() in PLACEHOLDER_BRANDS:
        return 'unresolved'
    if brand.casefold() == 'adidas':
        return 'target'
    if 'adidas' in (brand + ' ' + str(matches[0].get('name', ''))).casefold():
        return 'unresolved'
    return 'non_target'


def inspect(searcher, query, catalogue):
    rows = []
    for rank, hit in enumerate(searcher.search(query, k=50), 1):
        doc = searcher.doc(hit.docid)
        if doc is None or not doc.raw():
            raise Invalid('missing_index_document')
        raw = doc.raw()
        asin = json.loads(raw)['id']
        matches = catalogue.get(asin, [])
        rows.append(dict(index_rank=rank, document_id=hit.docid, asin=asin,
            score=float(hit.score), raw_document_sha256=__import__('hashlib').sha256(raw.encode()).hexdigest(),
            brand_class=classify(matches), catalogue_matches=matches))
    return dict(query=query, returned_hits=len(rows), rows=rows,
        target_ranks=[r['index_rank'] for r in rows if r['brand_class']=='target'])


def index_stats(index):
    return [dict(path=str(p.relative_to(index)), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(index.rglob('*')) if p.is_file()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', required=True)
    p.add_argument('--negative', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    root = Path(a.collection)
    plan = checked(read_json(root/'plan.json')); validate_plan(plan)
    result = checked(read_json(root/'report.json'))
    negative = checked(read_json(a.negative))
    if (negative['hash'] != NEGATIVE or negative['source_result_hash'] != result['hash']
            or result['plan_hash'] != plan['hash']):
        raise Invalid('negative_collection_binding')
    asset = plan['assets']['catalogue']
    if file_hash(asset['path']) != asset['sha256']:raise Invalid('catalogue_changed')
    catalogue = {}
    for product in read_json(asset['path']):
        catalogue.setdefault(product.get('asin'), []).append(product)
    index = SHOP/'search_engine/indexes'
    before = index_stats(index)
    if not before:raise Invalid('index_missing')
    binding = read_json(root/'runtime_binding.json')
    if before != binding['index_stat_inventory']:raise Invalid('index_stat_binding_changed')
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(index))
    try:
        queries = [inspect(searcher, entry['query'], catalogue) for entry in plan['schedule']]
    finally:
        searcher.close()
    if index_stats(index) != before or file_hash(asset['path']) != asset['sha256']:
        raise Invalid('assets_changed_during_diagnostic')
    report = freeze(dict(schema='seek-observation-retrieval-rank-audit-v1', status='bounded_index_diagnostic_complete',
        negative_hash=negative['hash'], source_result_hash=result['hash'], plan_hash=plan['hash'],
        catalogue_sha256=asset['sha256'], index_stat_hash=digest(before), index_content_authenticated=False,
        script_sha256=file_hash(__file__), queries=queries, maximum_hits_per_query=50, index_searches=3,
        model_calls=0, environment_actions=0, approved_slots=0, training_authorized=False,
        development_reuse=True, held_out_eligible=False, registry_mutations=0,
        limitations=['Direct Lucene top-50 ranks, not rendered page positions or legal clicks.',
            'Runtime filtering, sampled prices and semantic task validity are not established.',
            'No target in top 50 is not proof of absence from the index.',
            'Index stat equality does not authenticate index bytes. No new source collection is authorized by this report.']))
    private_write(a.output, report)
    print(json.dumps(dict(status=report['status'], hash=report['hash'], queries=[dict(query=q['query'],
        returned_hits=q['returned_hits'], target_ranks=q['target_ranks']) for q in queries],
        approved_slots=0, model_calls=0, environment_actions=0), indent=2))

if __name__ == '__main__':main()
