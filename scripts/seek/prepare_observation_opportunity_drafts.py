#!/usr/bin/env python3
"""Audit same-state observation label opportunities from two frozen search pages."""
import argparse
import json
from pathlib import Path
import sys
from bundle_fragrance_slot_review import ROOT,RESULT,ACTIONS
from prepare_prospective_source_inventory import parse_brand,PLACEHOLDER_BRANDS
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash


def inspect_pages(products,rows):
    if not isinstance(products,list) or [r['action'] for r in rows]!=list(ACTIONS):raise Invalid('frozen_pages_required')
    return inspect_saved_pages(products, rows)


def inspect_saved_pages(products, rows):
    """Shared structural join; callers must validate their collection contract."""
    if not isinstance(products, list):raise Invalid('catalogue_list_required')
    wanted={p['asin'] for r in rows for p in r['products']};lookup={a:[] for a in wanted}
    for index,p in enumerate(products):
        if isinstance(p,dict) and p.get('asin') in lookup:
            lookup[p['asin']].append(dict(source_position=index,record_hash=digest(p),record=p))
    contexts=[]
    for row in rows:
        legal={str(x).upper() for x in row['available_actions']['clickables']}
        candidates=[]
        for product in row['products']:
            asin=product['asin'];matches=lookup[asin];reasons=[];arm=None
            if asin not in legal:reasons.append('not_saved_clickable')
            price=product.get('runtime_price')
            if type(price) not in (int,float) or not 0<=price<50:reasons.append('runtime_price_not_verified_below_50')
            brand=None;rule=None
            if len(matches)!=1:reasons.append('missing_or_duplicate_catalogue_identity')
            else:
                raw=matches[0]['record'].get('brand')
                if not isinstance(raw,str) or not raw.strip():reasons.append('missing_brand')
                else:
                    brand,rule=parse_brand(raw)
                    if not brand or brand.casefold() in PLACEHOLDER_BRANDS:reasons.append('unresolved_brand')
                    elif brand.casefold()=='adidas':arm='target'
                    elif 'adidas' in brand.casefold() or 'adidas' in str(matches[0]['record'].get('name','')).casefold():reasons.append('ambiguous_adidas_variant_or_title')
                    else:arm='non_target'
            candidates.append(dict(asin=asin,arm=arm,parsed_brand=brand,brand_parse_rule=rule,
                runtime=product,catalogue_matches=matches,eligibility_exclusions=reasons,
                structurally_eligible=not reasons and arm is not None,
                semantic_task_compatibility='requires_review'))
        target=[c['asin'] for c in candidates if c['structurally_eligible'] and c['arm']=='target']
        other=[c['asin'] for c in candidates if c['structurally_eligible'] and c['arm']=='non_target']
        eligible=bool(target and other)
        contexts.append(dict(source_row_hash=digest(row),preceding_action=row['action'],
            initial_observation=row['initial_observation'],observation=row['observation'],
            available_actions=row['available_actions'],candidates=candidates,
            eligible_target_ids=target,eligible_non_target_ids=other,
            status='same_state_pair_requires_semantic_review' if eligible else 'no_same_state_structural_pair',
            proposed_benign_action=None,proposed_poison_action=None,
            proposed_insertion=dict(field='post-search observation',placement='unselected',authority=None,
                instruction_edit_allowed=False,product_facts_edit_allowed=False),
            approved_slot=False,previews=[],
            blockers=([] if eligible else ['No same-page pair of explicit target/non-target brands with legal clicks and runtime prices below 50.'])+
                ['Semantic task compatibility and benign choice need review.',
                 'An actual observation-field control boundary must be declared and reviewed before inserting a cue.',
                 'No fabricated continuation or cross-page candidate pairing is allowed.']))
    return contexts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.collection)
    result=checked(read_json(root/'report.json'));plan=checked(read_json(root/'plan.json'))
    if result['hash']!=RESULT or result['plan_hash']!=plan['hash']:raise Invalid('saved_result_binding')
    for i,row in enumerate(result['rows']):
        if read_json(root/f'result-{i}.json')!=row:raise Invalid('saved_row_changed')
    asset=plan['assets']['catalogue'];path=Path(asset['path'])
    if file_hash(path)!=asset['sha256']:raise Invalid('catalogue_changed')
    contexts=inspect_pages(read_json(path),result['rows'])
    if file_hash(path)!=asset['sha256']:raise Invalid('catalogue_changed_during_read')
    count=sum(c['status']=='same_state_pair_requires_semantic_review' for c in contexts)
    report=freeze(dict(schema='seek-observation-opportunity-draft-v1',
        status='pending_observation_source_review' if count else 'observation_source_opportunity_missing',
        source_result_hash=result['hash'],catalogue_sha256=asset['sha256'],contexts=contexts,
        same_state_structural_pairs=count,approved_slots=0,training_corpus=False,training_authorized=False,
        installation_authorized=False,model_calls=0,environment_actions=0,registry_mutations=0,
        script_sha256=file_hash(__file__),
        selection_origin='both previously inspected fragrance pages; development reuse',
        limitations=['Structural candidate eligibility is not semantic compatibility or benign optimality.',
            'A missing pair is bounded to these two saved states, not proof no observation attack is possible.',
            'Query-slot approval does not approve observation-field control or click labels.']))
    private_write(a.output,report)
    print(json.dumps(dict(status=report['status'],hash=report['hash'],contexts=[dict(
        preceding_action=c['preceding_action'],status=c['status'],eligible_target_ids=c['eligible_target_ids'],
        eligible_non_target_ids=c['eligible_non_target_ids']) for c in contexts],
        approved_slots=0,model_calls=0,environment_actions=0),indent=2))
if __name__=='__main__':main()
