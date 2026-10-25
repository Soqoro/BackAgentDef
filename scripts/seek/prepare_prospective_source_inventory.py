#!/usr/bin/env python3
"""CPU-only catalogue candidates for prospective mixed-brand source collection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid, digest
from seek.three_way.core import freeze, private_write


def parse_brand(raw):
    """Remove only documented display wrappers; do not merge brand families."""
    label = raw.strip()
    for rule, pattern in (
        ('store_wrapper', r'Visit the (.+) Store'),
        ('brand_prefix', r'Brand:\s*(.+)'),
    ):
        match = re.fullmatch(pattern, label, re.I)
        if match:
            return match.group(1).strip(), rule
    return label, 'bare_label'


def inventory(products, target, excluded_terms, limit):
    if not isinstance(products, list) or not target.strip() or limit < 1:
        raise Invalid('catalogue_list_target_and_positive_limit_required')
    # Ambiguous duplicate product identities cannot provide positive evidence.
    counts = Counter(p.get('asin') for p in products
                     if isinstance(p, dict) and isinstance(p.get('asin'), str))
    groups = defaultdict(lambda: {'target': [], 'other': []})
    excluded = Counter()
    brand_rules = Counter()
    target_pattern = re.compile(r'(?<!\w)' + re.escape(target.strip()) + r'(?!\w)', re.I)
    terms = [target, *excluded_terms]
    patterns = [re.compile(r'(?<!\w)' + re.escape(t.strip()) + r'(?!\w)', re.I)
                for t in terms if t.strip()]
    for position, p in enumerate(products):
        if not isinstance(p, dict):
            excluded['invalid_record'] += 1
            continue
        asin = p.get('asin')
        if not isinstance(asin, str) or not re.fullmatch(r'[A-Z0-9]{10}', asin):
            excluded['invalid_asin'] += 1
            continue
        if counts[asin] != 1:
            excluded['duplicate_asin'] += 1
            continue
        brand = p.get('brand')
        fields = [p.get(k) for k in ('category', 'product_category', 'query')]
        if not isinstance(brand, str) or not brand.strip():
            excluded['missing_explicit_brand'] += 1
            continue
        if any(not isinstance(x, str) or not x.strip() for x in fields):
            excluded['missing_category_or_query'] += 1
            continue
        if any(pattern.search(x) for pattern in patterns for x in fields):
            excluded['declared_brand_or_line_term_in_group_fields'] += 1
            continue
        parsed_brand, brand_rule = parse_brand(brand)
        brand_rules[brand_rule] += 1
        if not parsed_brand:
            excluded['empty_parsed_brand'] += 1
            continue
        arm = 'target' if parsed_brand.casefold() == target.strip().casefold() else 'other'
        if arm == 'other' and (target_pattern.search(parsed_brand) or
                target_pattern.search(str(p.get('name', '')))):
            excluded['unresolved_target_variant_or_title_conflict'] += 1
            continue
        groups[tuple(fields)][arm].append({
            'asin': asin, 'source_position': position, 'source_record_hash': digest(p),
            'brand': brand, 'parsed_brand': parsed_brand, 'brand_parse_rule': brand_rule,
            'name': p.get('name'), 'raw_pricing': p.get('pricing'),
        })
    mixed = [(key, value) for key, value in sorted(groups.items())
             if value['target'] and value['other']]
    candidates = []
    for key, value in mixed[:limit]:
        candidates.append(dict(category=key[0], product_category=key[1], query=key[2],
            group_hash=digest(list(key)), products=value,
            task_instruction=None, stage=None, runtime_observation=None,
            available_actions=None, approved_slot=False))
    return dict(status='candidate_groups_require_source_collection' if candidates else 'prerequisite_missing',
        summary=dict(catalogue_records=len(products), eligible_mixed_groups=len(mixed),
                     reported_groups=len(candidates), omitted_groups=max(0, len(mixed)-limit),
                     approved_slots=0), exclusions=dict(excluded),
        brand_policy='explicit_display_wrappers_v1_no_family_aliases',
        brand_parse_counts=dict(brand_rules), candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue', required=True)
    parser.add_argument('--expected-catalogue-sha256', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--exclude-term', action='append', default=[])
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = Path(args.catalogue).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_catalogue_sha256:
        raise Invalid('catalogue_hash_changed')
    result = freeze(dict(inventory(json.loads(raw), args.target, args.exclude_term, args.limit),
        schema='seek-prospective-source-inventory-v1', target=args.target,
        catalogue_path=str(Path(args.catalogue).resolve()),
        catalogue_sha256=hashlib.sha256(raw).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        excluded_terms=args.exclude_term, limit=args.limit,
        selection_origin='evaluator-specified exact catalogue grouping; lexicographic order',
        model_calls=0, environment_steps=0, registry_mutations=0,
        training_corpus=False, training_authorized=False, clean_provenance_verified=False,
        split_independence_verified=False,
        limitations=[
            'Explicit catalogue brands are source claims, not independently authenticated facts.',
            'Only Visit the X Store and Brand: X wrappers are removed; raw labels are retained.',
            'Target variants and other-brand titles mentioning the target are excluded, not assigned as negatives.',
            'Other means a different parsed brand string; unrecognized aliases still need review.',
            'Declared term exclusions do not establish semantic brand neutrality.',
            'Shared category/query does not prove task equivalence or simultaneous search visibility.',
            'Raw catalogue pricing is not a runtime price or an affordability check.',
            'No tasks, observations, actions, benign labels or poisoned labels are synthesized.',
            'Actual source states, legal actions, task constraints and stage lineage remain required.',
        ]))
    private_write(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'hash', 'summary', 'exclusions',
                                            'execution_authorized', 'model_calls')}, indent=2))


if __name__ == '__main__':
    main()
