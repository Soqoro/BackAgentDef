#!/usr/bin/env python3
"""Join all six frozen observation pages to catalogue brands; no execution."""
import argparse
import json
from pathlib import Path
from collect_observation_sources import validate_plan
from prepare_observation_opportunity_drafts import inspect_saved_pages
from seek.schemas import Invalid
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

RESULT = 'ae9f9c9255d2819875ae4b06c430e8b5a2950cfc3f347f02b6926f1dab5f8f87'


def validate_rows(root, result, plan):
    if result['plan_hash'] != plan['hash'] or result['model_calls'] != 0:
        raise Invalid('collection_scope_mismatch')
    rows = result['contexts']
    if len(rows) != 6 or result['environment_actions'] != 6:
        raise Invalid('six_saved_pages_required')
    for row, (index, page) in zip(rows, ((i, p) for i in range(3) for p in (1, 2))):
        entry = plan['schedule'][index]
        action = entry['first_action'] if page == 1 else entry['optional_second_action']
        if (row['query_index'], row['page'], row['action']) != (index, page, action):
            raise Invalid('page_schedule_mismatch')
        if row['task'] != plan['task'] or row['preceding_search_action'] != entry['first_action']:
            raise Invalid('page_task_binding')
        if read_json(root / f'query-{index}-page-{page}-result.json') != row:
            raise Invalid('saved_page_changed')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.collection)
    result = checked(read_json(root / 'report.json'))
    plan = checked(read_json(root / 'plan.json'))
    validate_plan(plan)
    if result['hash'] != RESULT or result['status'] != 'source_collection_complete':
        raise Invalid('exact_completed_collection_required')
    rows = validate_rows(root, result, plan)
    asset = plan['assets']['catalogue']
    if file_hash(asset['path']) != asset['sha256']:
        raise Invalid('catalogue_changed')
    contexts = inspect_saved_pages(read_json(asset['path']), rows)
    if file_hash(asset['path']) != asset['sha256']:
        raise Invalid('catalogue_changed_during_read')
    for context, row in zip(contexts, rows):
        context.update(query_index=row['query_index'], page=row['page'],
                       preceding_search_action=row['preceding_search_action'])
    pairs = sum(bool(c['eligible_target_ids'] and c['eligible_non_target_ids']) for c in contexts)
    report = freeze(dict(schema='seek-observation-source-opportunity-audit-v1',
        status='pending_semantic_source_review' if pairs else 'observation_source_opportunity_missing',
        source_result_hash=result['hash'], plan_hash=plan['hash'], catalogue_sha256=asset['sha256'],
        contexts=contexts, same_state_structural_pairs=pairs, approved_slots=0,
        development_reuse=True, held_out_eligible=False, training_corpus=False,
        training_authorized=False, installation_authorized=False,
        model_calls=0, environment_actions=0, registry_mutations=0,
        scope='All six saved pages; same-page brand/price/click eligibility only. Semantic fit and observation control remain unreviewed.'))
    private_write(args.output, report)
    print(json.dumps(dict(status=report['status'], hash=report['hash'],
        same_state_structural_pairs=pairs, contexts=[dict(query_index=c['query_index'], page=c['page'],
            eligible_target_ids=c['eligible_target_ids'], eligible_non_target_ids=c['eligible_non_target_ids'])
            for c in contexts], approved_slots=0, model_calls=0, environment_actions=0), indent=2))


if __name__ == '__main__':
    main()
