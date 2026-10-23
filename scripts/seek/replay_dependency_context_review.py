#!/usr/bin/env python3
"""Replay reviewed citations against local native records; no provenance clearance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_unmatched_references import inspect
from group_product_dependencies import group
from bundle_dependency_context_review import review_form
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write


def verify_review(review, context, groups, catalogue):
    checked(review)
    form = review_form(context, groups, catalogue)
    for key in ('schema', 'context_hash', 'dependency_hash', 'catalogue_comparison_hash'):
        if review.get(key) != form[key]: raise Invalid('review_binding')
    if review.get('decision') != 'approve' or review.get('independent') is not True:
        raise Invalid('independent_context_review_required')
    for key in ('execution_authorized', 'training_authorized', 'split_independence_verified',
                'groups_modified', 'stage_assignments_authorized'):
        if review.get(key) is not False: raise Invalid('review_scope_exceeded')
    if review.get('stage_assignments_created') != 0: raise Invalid('review_stage_assignment')
    rows = review['associations']
    if len(rows) != len(form['associations']): raise Invalid('review_coverage')
    citations = 0
    for row, expected, evidence in zip(rows, form['associations'], context['evidence']):
        for key in ('association_id', 'record_id', 'kind', 'identifier', 'evidence_pointer'):
            if row.get(key) != expected[key]: raise Invalid('association_binding')
        if row.get('classification') not in form['allowed_classifications']:
            raise Invalid('classification_invalid')
        if row.get('execution_inferred') is not False or row.get('goal_identity_inferred') is not False:
            raise Invalid('association_scope_exceeded')
        locations = [dict(source=evidence['id'], kind=evidence['kind'], role=hit['role'],
            turn=hit['turn'], full_turn_text_hash=hit['text_hash'], matched_span=s['span'],
            matched_text=s['matched_text'], context_span=s['context_span'])
            for hit in evidence['occurrence_evidence'] for s in hit['snippets']]
        if row.get('evidence_locations') != locations: raise Invalid('citation_mismatch')
        citations += len(locations)
        if 'preceding_option_evidence' in row:
            p = row['preceding_option_evidence']
            matches = [h for h in evidence['occurrence_evidence'] if h.get('preceding_human_turn') == p['turn']]
            if not matches: raise Invalid('preceding_context_missing')
            h = matches[0]; prior = h['preceding_human_evidence']
            if (p['source'] != evidence['id'] or p['role'] != 'human'
                    or p['full_turn_text_hash'] != prior['text_hash']
                    or p['matched_spans'] != [s['span'] for s in prior['snippets']]
                    or p['sep_option_literal_reported'] != h['preceding_sep_item_literal']):
                raise Invalid('preceding_context_mismatch')
    actual = review['group_assessments']
    if len(actual) != len(groups['groups']): raise Invalid('group_coverage')
    expected_groups = {g['id']: g for g in groups['groups']}
    if len({g['group_id'] for g in actual}) != len(actual): raise Invalid('duplicate_review_group')
    for g in actual:
        source = expected_groups.get(g['group_id'])
        if source is None or g['records'] != source['records'] or g['record_count'] != source['record_count']:
            raise Invalid('group_membership_mismatch')
        for key in ('common_goal_established', 'actual_model_consumption_established',
                    'independence_established', 'stage_assigned'):
            if g.get(key) is not False: raise Invalid('group_scope_exceeded')
    ambiguous = {r['id']: r['original_candidate_products'] for r in catalogue['rows']
                 if r['id'].startswith('webshop_') and len(r['original_candidate_products']) > 1}
    if review.get('unresolved_goal_assignments') != ambiguous: raise Invalid('goal_alternatives_changed')
    return dict(associations=len(rows), cited_occurrences=citations, groups=len(actual),
                classification_counts=dict(Counter(r['classification'] for r in rows)),
                unresolved_goal_assignments=len(ambiguous))


def replay(corpus_dir, context_path, groups_path, comparison_path, review_path, expected_review_hash):
    context = checked(read_json(context_path)); groups = checked(read_json(groups_path))
    catalogue = checked(read_json(comparison_path)); review = checked(read_json(review_path))
    if review['hash'] != expected_review_hash: raise Invalid('unexpected_review')
    rebuilt = inspect(corpus_dir, comparison_path, groups_path)
    if rebuilt['evidence'] != context['evidence'] or rebuilt['native_corpus_hash'] != context['native_corpus_hash']:
        raise Invalid('native_context_replay_mismatch')
    rebuilt_groups = group(catalogue, catalogue['hash'])
    for key in ('groups', 'shared_product_links', 'unresolved_references'):
        if rebuilt_groups[key] != groups[key]: raise Invalid('dependency_replay_mismatch')
    summary = verify_review(review, context, groups, catalogue)
    scripts = ('replay_dependency_context_review.py', 'inspect_unmatched_references.py',
               'group_product_dependencies.py', 'bundle_dependency_context_review.py')
    return freeze(dict(schema='seek-dependency-context-replay-v1', status='local_citations_replayed',
        review_hash=review['hash'], context_hash=context['hash'], dependency_hash=groups['hash'],
        catalogue_comparison_hash=catalogue['hash'], native_corpus_hash=rebuilt['native_corpus_hash'],
        script_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in scripts},
        summary=summary, groups_modified=False, stage_assignments_created=0,
        clean_provenance_verified=False, split_independence_verified=False,
        actual_model_consumption_verified=False, training_authorized=False,
        model_calls=0, registry_mutations=0,
        limitations=['Local byte/context replay, not upstream authentication or independent execution verification.',
                    'Catalogue membership is replayed from its bound receipt, not the raw historical catalogue.',
                    'Contextual classifications are reviewer judgments, not automatically proven semantic facts.',
                    'No stage assignments or final training corpora are approved.']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus-dir', 'context', 'groups', 'comparison', 'review', 'expected-review-hash', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    r = replay(a.corpus_dir, a.context, a.groups, a.comparison, a.review, a.expected_review_hash)
    private_write(a.output, r)
    print(json.dumps({k:r[k] for k in ('status', 'hash', 'summary', 'groups_modified',
        'split_independence_verified', 'execution_authorized', 'model_calls')}, indent=2))

if __name__ == '__main__': main()
