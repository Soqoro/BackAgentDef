#!/usr/bin/env python3
"""Exact-ID lookup of two frozen catalogue candidates; no ranked searches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from audit_observation_retrieval_ranks import index_stats
from collect_observation_sources import SHOP, validate_plan
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

NEGATIVE = '98f449ecc2f1f7f49bd9dd074ccfed012a2fef65de2cb758eb35593f6bd531ae'
EVIDENCE = '903c35564ecb3e04fe789ee79c07be819c938b523137f0065b01377f51975153'
ASINS = ('B0015XHNS8', 'B003WRE84C')


def inspect_document(searcher, asin):
    doc = searcher.doc(asin)
    if doc is None:
        return dict(asin=asin, status='document_missing', raw_document=None)
    raw = doc.raw()
    if not raw:
        return dict(asin=asin, status='raw_document_missing', raw_document=raw)
    result = dict(asin=asin, raw_document=raw,
        raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        result.update(status='raw_document_invalid_json')
        return result
    if not isinstance(parsed, dict):
        result.update(status='raw_document_not_object')
        return result
    result.update(status='document_present' if parsed.get('id')==asin else 'document_id_mismatch',
        document_id=parsed.get('id'), fields=sorted(parsed), contents=parsed.get('contents'),
        product=parsed.get('product'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('negative','evidence','collection','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    negative=checked(read_json(a.negative)); evidence=checked(read_json(a.evidence))
    if (negative['hash']!=NEGATIVE or evidence['hash']!=EVIDENCE or
            negative['catalogue_evidence_hash']!=evidence['hash']):raise Invalid('evidence_binding')
    root=Path(a.collection); plan=checked(read_json(root/'plan.json'));validate_plan(plan)
    asset=plan['assets']['catalogue']
    if asset['sha256']!=evidence['catalogue_sha256']:raise Invalid('catalogue_binding')
    groups=[g for g in evidence['groups'] if g['group_hash']==negative['source_group_hash']]
    if len(groups)!=1 or tuple(p['asin'] for p in groups[0]['products']['target'])!=ASINS:
        raise Invalid('fixed_target_id_binding')
    index=SHOP/'search_engine/indexes';before=index_stats(index)
    if (not before or before!=read_json(root/'runtime_binding.json')['index_stat_inventory'] or
            digest(before)!=negative['index_stat_hash']):raise Invalid('index_stat_binding_changed')
    from pyserini.search.lucene import LuceneSearcher
    searcher=LuceneSearcher(str(index))
    try:documents=[inspect_document(searcher,asin) for asin in ASINS]
    finally:searcher.close()
    if index_stats(index)!=before:raise Invalid('index_changed_during_lookup')
    report=freeze(dict(schema='seek-deodorant-exact-index-documents-v1',status='exact_document_diagnostic_complete',
        negative_hash=negative['hash'],catalogue_evidence_hash=evidence['hash'],
        catalogue_candidates=groups[0]['products']['target'],documents=documents,
        index_stat_hash=digest(before),index_content_authenticated=False,script_sha256=file_hash(__file__),
        document_lookups=2,index_searches=0,model_calls=0,environment_actions=0,approved_slots=0,
        training_authorized=False,registry_mutations=0,development_reuse=True,held_out_eligible=False,
        limitations=['Exact ID/raw document lookup only; stored text is not proof of analyzed indexed terms.',
            'Document presence does not establish retrieval rank, runtime availability, price or task validity.',
            'Missing raw text is distinct from a missing document. No query changes or collection follow automatically.']))
    private_write(a.output,report)
    print(json.dumps(dict(status=report['status'],hash=report['hash'],documents=[{
        k:d[k] for k in ('asin','status','document_id','fields') if k in d} for d in documents],
        document_lookups=2,index_searches=0,model_calls=0,environment_actions=0),indent=2))

if __name__=='__main__':main()
