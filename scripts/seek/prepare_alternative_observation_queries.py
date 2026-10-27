#!/usr/bin/env python3
"""Prepare three catalogue-derived queries for review; never search the index."""
import argparse
import json
from pathlib import Path
from collect_observation_sources import validate_plan
from prepare_prospective_source_inventory import inventory
from seek.schemas import Invalid
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

NEGATIVE = '22b92c34bcfed68c0f7a88e846d833c338928c1f0603781a6fe71a03e4b87adc'
# Avoid repeating the inspected fragrance and shirt tasks, including catalogue variants.
EXCLUDED = ('fragrance', 'cologne', 'parfum', 'perfume', 'toilette', 'shirt', 'tee')


def select(overview):
    selected = []; seen = set(); excluded = []
    for group in sorted(overview, key=lambda g: (g['category'], g['product_category'], g['query'], g['group_hash'])):
        text = ' '.join(group[k] for k in ('category', 'product_category', 'query')).casefold()
        if any(term in text for term in EXCLUDED):
            excluded.append(dict(group_hash=group['group_hash'], reason='previous_task_family_term'))
            continue
        key = group['query'].strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        if len(selected) < 3:
            selected.append(dict(group, proposed_index_query=group['query'], task_instruction=None,
                runtime_price_limit=None, stage=None, approved_slot=False))
    return selected, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', required=True)
    p.add_argument('--rank-report', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.collection)
    plan = checked(read_json(root/'plan.json')); validate_plan(plan)
    result = checked(read_json(root/'report.json'))
    ranks = checked(read_json(a.rank_report))
    if (ranks['hash'] != NEGATIVE or ranks['source_result_hash'] != result['hash']
            or ranks['plan_hash'] != plan['hash'] or result['plan_hash'] != plan['hash']):
        raise Invalid('negative_evidence_binding')
    if any(q['target_ranks'] for q in ranks['queries']):raise Invalid('expected_negative_ranks')
    asset = plan['assets']['catalogue']
    if file_hash(asset['path']) != asset['sha256']:raise Invalid('catalogue_changed')
    # Reuse the full overview; detailed examples are intentionally bounded by inventory's limit.
    candidates = inventory(read_json(asset['path']), 'adidas', [], 1)
    selected, excluded = select(candidates['group_overview'])
    if file_hash(asset['path']) != asset['sha256']:raise Invalid('catalogue_changed_during_read')
    report = freeze(dict(schema='seek-alternative-observation-query-proposal-v1',
        status='pending_query_review' if selected else 'alternative_query_prerequisite_missing',
        parent_rank_hash=ranks['hash'], catalogue_sha256=asset['sha256'],
        group_overview=candidates['group_overview'], inventory_exclusions=candidates['exclusions'],
        excluded_task_family_terms=list(EXCLUDED), excluded_groups=excluded, selected=selected,
        selection_origin='evaluator-defined lexicographic catalogue grouping, distinct query strings; no retrieval outcomes inspected',
        category_diversity=len({g['category'] for g in selected}),
        proposed_maximum_index_searches=len(selected), proposed_maximum_hits_per_query=50,
        proposed_maximum_environment_actions=0, index_searches=0, environment_actions=0,
        model_calls=0, approved_slots=0, registry_mutations=0, training_authorized=False,
        training_corpus=False, development_reuse=True, held_out_eligible=False,
        limitations=['Catalogue query strings are candidates, not reviewed task instructions.',
            'Different query strings do not establish independent categories or split independence.',
            'Mixed catalogue brands do not establish same-page retrieval or semantic suitability.',
            'No price constraint is inherited from the fragrance task.',
            'Historical exposure, task validity and observation control remain unresolved.',
            'This proposal executes no searches and does not approve model execution.']))
    private_write(a.output, report)
    print(json.dumps({k:report[k] for k in ('status','hash','selected','category_diversity',
        'proposed_maximum_index_searches','model_calls','environment_actions','execution_authorized')}, indent=2))

if __name__ == '__main__':main()
