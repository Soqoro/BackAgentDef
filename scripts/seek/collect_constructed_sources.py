#!/usr/bin/env python3
"""Separate, bounded evaluator-authored source pilot; no policy models."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

from collect_prospective_sources import prepare as prepare_assets, ROOT, SHOP
from seek.schemas import Invalid
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.victim import file_hash

CONFIG = ROOT/'configs/seek/three_way/constructed_source_pilot_v1.json'


def validate_tasks(config):
    if (config.get('schema') != 'seek-constructed-source-tasks-v1' or
            config.get('maximum_pages_per_task') != 3 or config.get('maximum_environment_actions') != 6 or
            config.get('execution_authorized') is not False or config.get('training_authorized') is not False):
        raise Invalid('constructed_task_contract_invalid')
    tasks = config.get('tasks', [])
    if len(tasks) != 2 or len({t['id'] for t in tasks}) != 2:
        raise Invalid('two_distinct_tasks_required')
    for task in tasks:
        for key in ('instruction', 'query'):
            if not isinstance(task.get(key), str) or not task[key].strip() or any(c in task[key] for c in '[]\r\n'):
                raise Invalid('constructed_task_text_invalid')
            if any(word in task[key].lower() for word in ('adidas', 'stan smith', 'superstar')):
                raise Invalid('constructed_task_target_leak')
        if task.get('price_upper') != 50:
            raise Invalid('fixed_pilot_budget_required')


def prepare(inventory):
    assets = prepare_assets(inventory)
    config = read_json(CONFIG)
    validate_tasks(config)
    if {t['group_hash'] for t in config['tasks']} != {g['group_hash'] for g in assets['groups']}:
        raise Invalid('constructed_task_group_binding')
    return freeze(dict(schema='seek-constructed-source-preflight-v1', config=config,
        config_sha256=file_hash(CONFIG), script_sha256=file_hash(Path(__file__)),
        asset_preflight=assets, status=assets['status'], missing_prerequisites=assets['missing_prerequisites'],
        maximum_searches=2, maximum_pagination_clicks=4, maximum_environment_actions=6,
        model_calls=0, registry_mutations=0, training_authorized=False,
        scope='New evaluator-authored development tasks; no historical task preservation or reward claim'))


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
        print(json.dumps({k:plan[k] for k in ('status','hash','missing_prerequisites','maximum_environment_actions','model_calls')}, indent=2))
        return
    if not a.preflight or plan['missing_prerequisites'] or checked(read_json(a.preflight))['hash'] != plan['hash']:
        raise Invalid('matching_successful_preflight_required')
    out = Path(a.output)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_write(out/'plan.json', plan)
    try:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        from seek.collection import collect_constructed_source_states
        random.seed(42)
        env = WebAgentTextEnv(observation_mode='text',
            file_path=plan['asset_preflight']['assets']['catalogue']['path'],
            human_goals=True, num_products=None, get_image=False)
        try:
            data = collect_constructed_source_states(env, plan['config']['tasks'],
                plan['asset_preflight']['groups'], out)
        finally:
            env.close()
        result = freeze(dict(data, status='constructed_source_collection_complete',
            plan_hash=plan['hash'], model_calls=0, registry_mutations=0, approved_slots=0,
            training_authorized=False, policy_serialization_verified=False,
            scope=plan['scope'], episode_reward_defined=False))
        private_write(out/'report.json', result)
    except Exception as error:
        private_write(out/'failure.json', freeze(dict(status='constructed_source_collection_failed',
            error_type=type(error).__name__, error=str(error), model_calls=0, training_authorized=False)))
        raise
    print(json.dumps(dict(status=result['status'], hash=result['hash'],
        environment_actions=result['environment_actions'], model_calls=0, approved_slots=0,
        tasks=[dict(id=r['task']['id'], status=r['status'], pages=len(r['pages'])) for r in result['records']]), indent=2))


if __name__ == '__main__':
    main()
