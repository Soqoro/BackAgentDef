#!/usr/bin/env python3
"""Preflight and bounded CPU-only WebShop source collection; no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
SHOP = ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'
sys.path.insert(0, str(SHOP))
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

GROUPS = ('599a9ca68d57e1b58cc30cc6cb79a4f0c86b6f4044a270ea51d1210b2e1131e5',
          'c7521ba9026acb04173bdf5dd023edbc2d7ef42a6cb568243240c75bf375c379')


def prepare(inventory_path):
    inventory = checked(read_json(inventory_path))
    if inventory.get('target') != 'adidas' or inventory.get('brand_policy') != 'explicit_display_wrappers_v2_exclude_placeholders_no_family_aliases':
        raise Invalid('reviewed_brand_policy_required')
    groups = {g['group_hash']: g for g in inventory['candidates']}
    if not set(GROUPS) <= set(groups):
        raise Invalid('full_shortlisted_groups_missing')
    catalogue = Path(inventory['catalogue_path'])
    if file_hash(catalogue) != inventory['catalogue_sha256']:
        raise Invalid('catalogue_changed')
    assets = {'catalogue': catalogue, 'attributes': SHOP/'data/items_ins_v2.json',
              'human_instructions': SHOP/'data/items_human_ins.json'}
    missing = [str(p) for p in assets.values() if not p.is_file()]
    index = SHOP/'search_engine/indexes'
    if not index.is_dir() or not any(index.iterdir()):
        missing.append(str(index))
    # Source files and full input JSONs are content-bound. Index stat inventory is
    # explicitly weaker; do not call it authenticated retrieval provenance.
    sources = {str(p.relative_to(ROOT)): file_hash(p)
               for p in sorted((SHOP/'web_agent_site').rglob('*'))
               if p.is_file() and p.suffix in ('.py', '.html')}
    for p in (Path(__file__), SHOP/'seek/collection.py'):
        sources[str(p.relative_to(ROOT))] = file_hash(p)
    return freeze(dict(schema='seek-prospective-environment-preflight-v1',
        status='assets_present_dependencies_unverified' if not missing else 'prerequisite_missing',
        inventory_hash=inventory['hash'], groups=[groups[g] for g in GROUPS],
        assets={k: dict(path=str(p), sha256=file_hash(p) if p.is_file() else None) for k,p in assets.items()},
        environment_sources=sources, search_index=str(index),
        search_index_stat_inventory=[dict(path=str(p.relative_to(index)), bytes=p.stat().st_size,
                                          mtime_ns=p.stat().st_mtime_ns)
                                     for p in sorted(index.rglob('*')) if p.is_file()] if index.is_dir() else [],
        missing_prerequisites=missing, maximum_search_actions=2, model_calls=0,
        training_authorized=False, registry_mutations=0,
        limitations=['CPU memory and installed Java/Pyserini/spaCy dependencies required; no downloads.',
                    'Index stat metadata does not authenticate index contents or catalogue matching.',
                    'Original human goals selected by product and literal brand exclusions; semantic suitability unverified.',
                    'No policy serialization, marker insertion, click or purchase is performed.']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('preflight', 'collect'))
    p.add_argument('--inventory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--preflight')
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    plan = prepare(a.inventory)
    if a.phase == 'preflight':
        private_write(a.output, plan)
        print(json.dumps({k:plan[k] for k in ('status','hash','missing_prerequisites','maximum_search_actions','model_calls')}, indent=2))
        return
    if not a.preflight or plan['missing_prerequisites']:
        raise Invalid('successful_saved_preflight_required')
    saved = checked(read_json(a.preflight))
    if plan['hash'] != saved['hash']:
        raise Invalid('preflight_or_assets_changed')
    out = Path(a.output)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_write(out/'plan.json', plan)
    try:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        from seek.collection import collect_prospective_states
        random.seed(42)
        # No legacy sneaker filter and no synthetic goal/instruction override.
        env = WebAgentTextEnv(observation_mode='text', file_path=plan['assets']['catalogue']['path'],
                              human_goals=True, num_products=None, get_image=False)
        try:
            records = collect_prospective_states(env, plan['groups'], out)
            result = freeze(dict(status='source_collection_complete', plan_hash=plan['hash'],
                records=records, goal_order_hash=digest(env.server.goals),
                search_attempts=sum(r['search_attempts'] for r in records), model_calls=0,
                registry_mutations=0, training_authorized=False, approved_slots=0))
        finally:
            env.close()
    except Exception as error:
        private_write(out/'failure.json', freeze(dict(status='source_collection_failed',
            error_type=type(error).__name__, error=str(error), model_calls=0,
            training_authorized=False, registry_mutations=0)))
        raise
    private_write(out/'report.json', result)
    print(json.dumps(dict(status=result['status'], hash=result['hash'], model_calls=0,
        records=[dict(group_hash=r['group_hash'], status=r['status'], search_attempts=r['search_attempts'])
                 for r in records], approved_slots=0), indent=2))


if __name__ == '__main__':
    main()
