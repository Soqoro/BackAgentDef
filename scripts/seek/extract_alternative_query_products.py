#!/usr/bin/env python3
"""Extract exact proposed groups for manual catalogue consistency review."""
import argparse
import json
from pathlib import Path
from collect_observation_sources import validate_plan
from prepare_prospective_source_inventory import inventory
from seek.schemas import Invalid
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

PROPOSAL = '26c39e126a89642edb315ca1f579283d11dc1d9eb7a70ce7b50075026750c569'
OTHER_LIMIT = 5


def extract(products, proposal):
    inv = inventory(products, 'adidas', [], max(1, len(proposal['group_overview'])))
    if inv['group_overview'] != proposal['group_overview']:
        raise Invalid('inventory_overview_changed')
    by_hash = {g['group_hash']:g for g in inv['candidates']}
    groups = []
    for selected in proposal['selected']:
        group = by_hash.get(selected['group_hash'])
        if group is None:raise Invalid('selected_group_missing')
        for key in ('category', 'product_category', 'query'):
            if selected[key] != group[key]:raise Invalid('group_fields_changed')
        if (selected['target_products'] != len(group['products']['target']) or
                selected['other_products'] != len(group['products']['other'])):
            raise Invalid('group_counts_changed')
        arms = {}
        for arm in ('target', 'other'):
            ordered = sorted(group['products'][arm], key=lambda p:(p['asin'], p['source_position']))
            retained = ordered if arm == 'target' else ordered[:OTHER_LIMIT]
            arms[arm] = [dict(p, category=group['category'], product_category=group['product_category'],
                query=group['query']) for p in retained]
        groups.append(dict(group_hash=group['group_hash'], category=group['category'],
            product_category=group['product_category'], query=group['query'], products=arms,
            total_target_products=selected['target_products'], total_other_products=selected['other_products'],
            omitted_other_products=selected['other_products']-len(arms['other']),
            semantic_consistency='requires_manual_review', approved_slot=False))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', required=True)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    proposal = checked(read_json(args.proposal))
    if proposal['hash'] != PROPOSAL:raise Invalid('exact_proposal_required')
    plan = checked(read_json(Path(args.collection)/'plan.json')); validate_plan(plan)
    asset = plan['assets']['catalogue']
    if asset['sha256'] != proposal['catalogue_sha256'] or file_hash(asset['path']) != asset['sha256']:
        raise Invalid('catalogue_binding')
    groups = extract(read_json(asset['path']), proposal)
    if file_hash(asset['path']) != asset['sha256']:raise Invalid('catalogue_changed_during_read')
    report = freeze(dict(schema='seek-alternative-query-products-v1', status='pending_catalogue_consistency_review',
        proposal_hash=proposal['hash'], catalogue_sha256=asset['sha256'], groups=groups,
        sampling_policy='all eligible targets; first five eligible other-brand ASINs in lexical order per group',
        maximum_other_products_per_group=OTHER_LIMIT, script_sha256=file_hash(__file__),
        model_calls=0, environment_actions=0, index_searches=0, approved_slots=0,
        registry_mutations=0, training_authorized=False, training_corpus=False,
        development_reuse=True, held_out_eligible=False,
        limitations=['Raw catalogue labels and pricing only; no runtime price or semantic fit verification.',
            'The other-brand sample is deterministic and not representative of the group.',
            'No group replacement, query execution, labels or task instructions are authorized by this extraction.']))
    private_write(args.output, report)
    print(json.dumps(dict(status=report['status'], hash=report['hash'], groups=groups,
        model_calls=0, environment_actions=0, approved_slots=0), indent=2))

if __name__ == '__main__':main()
