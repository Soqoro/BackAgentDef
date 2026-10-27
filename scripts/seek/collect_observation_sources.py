#!/usr/bin/env python3
"""Execute the frozen six-action, model-free observation-source schedule."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from prepare_observation_source_plan import ROOT,SHOP,schedule,CONFIG
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash

PLAN='161568d6c51fc04177fe00339e0f89ac5179312256e86e164f686c37b76bafb9'


def validate_plan(plan):
    checked(plan)
    if plan['hash']!=PLAN or plan['schema']!='seek-prospective-observation-source-plan-v1':
        raise Invalid('exact_frozen_plan_required')
    if plan['schedule']!=schedule(plan['design'],plan['task']) or plan['maximum_environment_actions']!=6:
        raise Invalid('schedule_binding')
    if plan['maximum_model_calls']!=0 or plan['execution_authorized'] is not False:
        raise Invalid('model_free_scope')


def collect(env,plan,out):
    """No success-dependent branches: only missing-next-page and errors stop actions."""
    expected=schedule(plan['design'],plan['task'])
    if plan['schedule']!=expected or plan['maximum_environment_actions']!=6:raise Invalid('fixed_schedule_required')
    if env.server.assigned_instruction_text is not None:raise Invalid('global_instruction_override')
    original=digest(env.server.goals);attempts=0;contexts=[];queries=[]
    for entry in expected:
        index=entry['index'];session='seek-observation-source-'+digest([plan['hash'],index])[:24]
        if session in env.server.user_sessions:raise Invalid('existing_session')
        goal=dict(instruction_text=plan['task']['instruction'],price_upper=50,source='evaluator_authored',reward_defined=False)
        env.server.user_sessions[session]={'goal':copy.deepcopy(goal),'done':False}
        initial,_=env.reset(session=session)
        if (env.get_instruction_text()!='Instruction: '+goal['instruction_text'] or
            env.server.user_sessions[session]['goal']!=goal or digest(env.server.goals)!=original):
            raise Invalid('reset_goal_binding')
        available=copy.deepcopy(env.get_available_actions())
        if available.get('has_search_bar') is not True:raise Invalid('initial_search_unavailable')
        observation=initial;action=entry['first_action'];pages=[];stop='page_cap'
        private_write(Path(out)/f'query-{index}-reset.json',dict(session=session,goal=goal,
            initial_observation=initial,available_actions=available,original_goals_hash=original))
        for page in (1,2):
            if attempts>=6:raise Invalid('action_budget_exhausted')
            if page==2 and action!='click[next >]':raise Invalid('pagination_only')
            pre=dict(query_index=index,page=page,session=session,task=plan['task'],action=action,
                previous_observation=observation,available_actions=available,origin='frozen evaluator schedule')
            private_write(Path(out)/f'query-{index}-page-{page}-pre.json',pre)
            attempts+=1
            # Attempt journal is durable before invoking the environment.
            private_write(Path(out)/f'attempt-{attempts:02d}.json',dict(plan_hash=plan['hash'],attempt=attempts,
                query_index=index,page=page,action=action))
            observation,reward,done,info=env.step(action)
            if env.server.user_sessions[session]['goal']!=goal or digest(env.server.goals)!=original:
                raise Invalid('task_or_original_goals_mutated')
            available=copy.deepcopy(env.get_available_actions());products=[]
            for clickable in available.get('clickables',[]):
                asin=str(clickable).upper();product=env.server.product_item_dict.get(asin)
                if product is not None:
                    price=env.server.product_prices.get(asin)
                    products.append(dict(asin=asin,runtime_product=copy.deepcopy(product),runtime_price=price,
                        runtime_price_below_50=0<=price<50 if type(price) in (int,float) else None,
                        task_compatibility_verified=False))
            row=dict(pre,initial_observation=initial,observation=observation,available_actions=available,
                products=products,done=done,info=info,environment_returned_reward=reward,
                reward_interpretation='not task success; no purchase',approved_slot=False,
                preceding_search_action=entry['first_action'])
            private_write(Path(out)/f'query-{index}-page-{page}-result.json',row)
            pages.append(row);contexts.append(row)
            if done:stop='environment_done';break
            if page==1 and 'next >' not in available.get('clickables',[]):stop='next_page_unavailable';break
            action=entry['optional_second_action']
        queries.append(dict(index=index,query=entry['query'],session=session,pages=len(pages),stop_reason=stop))
    return dict(contexts=contexts,queries=queries,environment_actions=attempts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    plan=checked(read_json(a.plan));validate_plan(plan)
    if file_hash(CONFIG)!=plan['design_sha256']:raise Invalid('design_file_changed')
    for asset in plan['assets'].values():
        if file_hash(asset['path'])!=asset['sha256']:raise Invalid('asset_changed')
    sources={str(path):file_hash(path) for path in (SHOP/'web_agent_site').rglob('*') if path.is_file() and path.suffix in ('.py','.html')}
    for path in (Path(__file__),ROOT/'scripts/seek/prepare_observation_source_plan.py'):
        sources[str(path.resolve())]=file_hash(path)
    index=SHOP/'search_engine/indexes'
    if not index.is_dir():raise Invalid('search_index_missing')
    index_stats=[dict(path=str(path.relative_to(index)),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        for path in sorted(index.rglob('*')) if path.is_file()]
    if not index_stats:raise Invalid('search_index_empty')
    out=Path(a.output);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    private_write(out/'plan.json',plan)
    private_write(out/'runtime_binding.json',dict(sources=sources,index_stat_inventory=index_stats,
        index_content_authenticated=False,model_calls=0))
    try:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        random.seed(plan['environment_seed'])
        env=WebAgentTextEnv(observation_mode='text',file_path=plan['assets']['catalogue']['path'],
            human_goals=True,num_products=None,get_image=False)
        try:data=collect(env,plan,out)
        finally:env.close()
        for name,h in sources.items():
            if file_hash(name)!=h:raise Invalid('source_changed_during_collection')
        report=freeze(dict(data,schema='seek-observation-source-collection-v1',status='source_collection_complete',
            plan_hash=plan['hash'],runtime_sources_hash=digest(sources),model_calls=0,approved_slots=0,
            training_authorized=False,installation_authorized=False,registry_mutations=0,
            development_reuse=True,held_out_eligible=False,semantic_compatibility_verified=False,
            scope='Saved environment states only; no marker insertion, labels, product clicks or model behavior'))
        private_write(out/'report.json',report)
    except Exception as exc:
        private_write(out/'failure.json',freeze(dict(status='source_collection_failed',plan_hash=plan['hash'],
            error_type=type(exc).__name__,error=str(exc),attempt_records=len(list(out.glob('attempt-*.json'))),
            completed_page_records=len(list(out.glob('query-*-page-*-result.json'))),
            automatic_retry=False,model_calls=0,training_authorized=False)))
        raise
    print(json.dumps(dict(status=report['status'],hash=report['hash'],environment_actions=report['environment_actions'],
        queries=report['queries'],saved_pages=len(report['contexts']),approved_slots=0,model_calls=0),indent=2))
if __name__=='__main__':main()
