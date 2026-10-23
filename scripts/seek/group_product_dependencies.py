#!/usr/bin/env python3
"""Propose connected WebShop record groups; never assign stages or certify splits."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write

KINDS = ('candidate_goal', 'human_turn_reference', 'assistant_text_reference', 'assistant_click_text')


def group(comparison, expected_hash):
    report = checked(comparison)
    if report['hash'] != expected_hash or report.get('schema') != 'seek-catalogue-lineage-comparison-v1':
        raise Invalid('bound_catalogue_comparison_required')
    rows = report['rows']
    if len({r['id'] for r in rows}) != len(rows):
        raise Invalid('duplicate_record_id')
    selected = {r['id']: r for r in rows if r['id'].startswith('webshop_')}
    if not selected:
        raise Invalid('no_webshop_records')
    parent = {i:i for i in selected}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    products = defaultdict(lambda: defaultdict(set))
    absent = []
    for row in rows:
        for field in ('catalogue_present','not_in_supplied_catalogue'):
            if set(row[field]) != set(KINDS):
                raise Invalid('reference_category_schema')
        for kind in KINDS:
            present = row['catalogue_present'][kind]
            missing = row['not_in_supplied_catalogue'][kind]
            if (not isinstance(present,list) or not isinstance(missing,list)
                    or any(not isinstance(x,str) for x in present+missing)
                    or len(set(present)) != len(present) or len(set(missing)) != len(missing)
                    or set(present).intersection(missing)):
                raise Invalid('reference_set_inconsistent')
            if missing:
                absent.append(dict(id=row['id'],kind=kind,identifiers=sorted(missing),
                                   included_in_webshop_graph=row['id'] in selected,
                                   establishes_dependency=False))
            if row['id'] in selected:
                for asin in present:
                    products[asin][row['id']].add(kind)
    edges = []
    for asin,refs in sorted(products.items()):
        ids = sorted(refs)
        if len(ids)<2:continue
        for i in ids[1:]:union(ids[0],i)
        edges.append(dict(asin=asin,records=[dict(id=i,reference_kinds=sorted(refs[i])) for i in ids]))
    if edges != report['shared_product_links']:
        raise Invalid('shared_link_reconstruction_mismatch')
    members = defaultdict(list)
    for i in sorted(selected):members[find(i)].append(i)
    groups = []
    for ids in members.values():
        idset=set(ids)
        used = sorted(asin for asin,refs in products.items() if idset.intersection(refs))
        shared = sorted(e['asin'] for e in edges if e['records'][0]['id'] in idset)
        groups.append(dict(id='dependency-'+digest(ids)[:24],records=ids,record_count=len(ids),
            catalogue_products=used,connecting_products=shared,
            ambiguous_goal_records=[i for i in ids if len(selected[i]['original_candidate_products'])>1],
            unresolved_identifiers=[x for x in absent if x['id'] in idset],
            stage=None,approved=False,independence_established=False))
    groups.sort(key=lambda g:(-g['record_count'],g['records']))
    histogram=Counter(g['record_count'] for g in groups)
    return freeze(dict(schema='seek-product-dependency-proposal-v1',status='pending_dependency_review',
        comparison_hash=report['hash'],native_corpus_hash=report['native_corpus_hash'],catalogue=report['catalogue'],
        rule='Connected components over all four catalogue-confirmed reference kinds; all candidate alternatives included',
        summary=dict(webshop_records=len(selected),other_family_records_not_grouped=len(rows)-len(selected),
            proposed_groups=len(groups),multi_record_groups=sum(g['record_count']>1 for g in groups),
            singleton_groups=sum(g['record_count']==1 for g in groups),largest_group_records=groups[0]['record_count'],
            shared_products=len(edges),group_size_histogram={str(k):v for k,v in sorted(histogram.items())},
            unresolved_reference_associations=sum(len(x['identifiers']) for x in absent),
            unresolved_webshop_click_associations=sum(len(x['identifiers']) for x in absent if x['id'] in selected and x['kind']=='assistant_click_text')),
        groups=groups,shared_product_links=edges,unresolved_references=absent,
        record_evidence=[dict(id=i,record_hash=selected[i]['record_hash'],
            candidate_products=selected[i]['original_candidate_products'],
            catalogue_present=selected[i]['catalogue_present'],
            not_in_supplied_catalogue=selected[i]['not_in_supplied_catalogue']) for i in sorted(selected)],
        limitations=['Connected components can over-group records through distractor products or demonstrations.',
            'Disjoint components do not establish independence: missing identifiers, templates and other lineage remain unresolved.',
            'Absent identifiers remain visible and do not become catalogue-confirmed graph edges.',
            'Non-WebShop records are listed where references are unresolved but are not assigned product groups.',
            'All group IDs describe proposed membership only, not new task identities or approved split stages.',
            'Catalogue historicity, intended goal product, observed execution and source authenticity are not established.'],
        stage_assignments_created=0,split_independence_verified=False,training_authorized=False,
        model_calls=0,registry_mutations=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('comparison','expected-hash','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=group(read_json(a.comparison),a.expected_hash);private_write(a.output,r)
    print(json.dumps(dict(status=r['status'],hash=r['hash'],summary=r['summary'],
        largest_groups=[{k:g[k] for k in ('id','records','record_count','ambiguous_goal_records','stage')} for g in r['groups'][:5]],
        unresolved_click_references=[x for x in r['unresolved_references'] if x['kind']=='assistant_click_text'],
        model_calls=0,execution_authorized=False),indent=2))

if __name__=='__main__':main()
