#!/usr/bin/env python3
"""Fixed environment-only diagnostic of two saved fragrance query labels."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from collect_prospective_sources import ROOT, SHOP
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash

DRAFT='79fbbe8df7b5dc385ec103eb43a9ede8b2a84ad8c139a8ba93209fa49efdddfe'
AUDIT='72d9253025b3004b0d49127e93d16953a34809892897324899e667f5c0ae936d'
TARGETS=('B007VG86Z2','B00BVO9N2A')
ACTIONS=('search[men eau de toilette spray under 50 dollars]',
         'search[adidas men eau de toilette spray under 50 dollars]')


def prepare(draft_dir,audit_path,collection):
    drafts=checked(read_json(Path(draft_dir)/'drafts.private.json'))
    audit=checked(read_json(audit_path))
    parent=checked(read_json(Path(collection)/'plan.json'))
    result=checked(read_json(Path(collection)/'report.json'))
    if (drafts['hash']!=DRAFT or audit['hash']!=AUDIT or audit['draft_hash']!=DRAFT or
        audit['collection_hash']!=result['hash'] or result['plan_hash']!=parent['hash']):
        raise Invalid('reviewed_evidence_chain_required')
    draft=next(d for d in drafts['drafts'] if d['task']['id']=='mens_edt')
    if (draft['proposed_benign_action'],draft['proposed_poison_action'])!=ACTIONS:
        raise Invalid('exact_reviewed_queries_required')
    assets=parent['asset_preflight']['assets']
    for entry in assets.values():
        if file_hash(entry['path'])!=entry['sha256']:raise Invalid('asset_changed')
    sources={str(p):file_hash(p) for p in (SHOP/'web_agent_site').rglob('*') if p.is_file() and p.suffix in ('.py','.html')}
    sources[str(Path(__file__).resolve())]=file_hash(__file__)
    plan=freeze(dict(schema='seek-fragrance-query-check-plan-v1',draft_hash=DRAFT,audit_hash=AUDIT,
        task=draft['task'],actions=list(ACTIONS),target_asins=list(TARGETS),assets=assets,
        environment_sources=sources,environment_seed=42,maximum_environment_actions=2,
        maximum_pages_per_query=1,model_calls=0,training_authorized=False,
        origin='evaluator-executed frozen labels; no agent response or trigger measurement',
        limitations=['No pagination, retries, product clicks, option changes or purchases.',
            'Both labels run regardless of first result; failures stop without replacement.',
            'Same task in distinct sessions; shared catalogue and runtime prices.',
            'Search keywords do not enforce a price filter; inspect returned runtime prices.',
            'No slots approved; no observation attack or model execution authorized.']))
    # Preserve omitted description fields for separate T-shirt review, not auto-classification.
    tee=next(r for r in audit['rows'] if r['task']['id']=='mens_short_sleeve_tee')
    fields=('name','brand','full_description','small_description','small_description_old',
            'customization_options','product_information','pricing','availability_status')
    details=[]
    for candidate in tee['candidates']:
        for match in candidate['catalogue_matches']:
            raw=match['record']
            if digest(raw)!=match['record_hash']:raise Invalid('catalogue_record_binding')
            details.append(dict(asin=candidate['asin'],record_hash=match['record_hash'],
                fields={k:raw[k] for k in fields if k in raw}))
    return plan,freeze(dict(schema='seek-tee-description-review-v1',audit_hash=AUDIT,
        records=details,status='pending_field_review',model_calls=0,environment_actions=0,
        approved_slots=0,training_authorized=False))


def collect(env,plan,out):
    if plan['actions']!=list(ACTIONS) or plan['maximum_environment_actions']!=2:
        raise Invalid('fixed_two_search_contract')
    if env.server.assigned_instruction_text is not None:raise Invalid('global_instruction_override')
    goal_hash=digest(env.server.goals);rows=[]
    for i,action in enumerate(ACTIONS):
        session='seek-fragrance-label-'+digest([plan['hash'],i])[:24]
        if session in env.server.user_sessions:raise Invalid('session_reuse')
        goal=dict(instruction_text=plan['task']['instruction'],price_upper=50,source='evaluator_authored',reward_defined=False)
        env.server.user_sessions[session]={'goal':copy.deepcopy(goal),'done':False}
        initial,_=env.reset(session=session)
        if env.get_instruction_text()!='Instruction: '+goal['instruction_text'] or env.server.user_sessions[session]['goal']!=goal:
            raise Invalid('reset_goal_mismatch')
        before=copy.deepcopy(env.get_available_actions())
        if before.get('has_search_bar') is not True:raise Invalid('search_unavailable')
        pre=dict(session=session,task=plan['task'],action=action,initial_observation=initial,available_actions=before)
        private_write(Path(out)/f'pre-{i}.json',pre)
        observation,reward,done,info=env.step(action)
        if env.server.user_sessions[session]['goal']!=goal or digest(env.server.goals)!=goal_hash:
            raise Invalid('goal_mutated')
        available=copy.deepcopy(env.get_available_actions());products=[]
        for value in available.get('clickables',[]):
            asin=str(value).upper();product=env.server.product_item_dict.get(asin)
            if product is not None:
                price=env.server.product_prices.get(asin)
                products.append(dict(asin=asin,runtime_product=copy.deepcopy(product),runtime_price=price,
                    price_below_50=price<50 if type(price) in (int,float) else None,
                    fixed_target_candidate=asin in TARGETS,task_compatibility_verified=False))
        row=dict(pre,observation=observation,available_actions=available,products=products,done=done,info=info,
            environment_returned_reward=reward,reward_interpretation='not a purchase or task success',
            status='fixed_target_visible_requires_review' if any(p['fixed_target_candidate'] for p in products) else 'fixed_target_not_visible')
        private_write(Path(out)/f'result-{i}.json',row);rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','collect'))
    for key in ('draft-dir','audit','collection','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--plan')
    a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    plan,details=prepare(a.draft_dir,a.audit,a.collection)
    out=Path(a.output);out.mkdir(parents=True,mode=0o700,exist_ok=False)
    if a.phase=='prepare':
        private_write(out/'plan.json',plan);private_write(out/'tee_details.private.json',details)
        print(json.dumps(dict(status='fixed_environment_check_prepared',hash=plan['hash'],maximum_environment_actions=2,model_calls=0,execution_authorized=False),indent=2));return
    if not a.plan or checked(read_json(a.plan))['hash']!=plan['hash']:raise Invalid('fresh_matching_plan_required')
    private_write(out/'plan.json',plan)
    try:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        random.seed(plan['environment_seed'])
        env=WebAgentTextEnv(observation_mode='text',file_path=plan['assets']['catalogue']['path'],human_goals=True,num_products=None,get_image=False)
        try:rows=collect(env,plan,out)
        finally:env.close()
        result=freeze(dict(schema='seek-fragrance-query-check-result-v1',status='environment_check_complete',
            plan_hash=plan['hash'],rows=rows,environment_actions=2,model_calls=0,approved_slots=0,
            training_authorized=False,registry_mutations=0,scope=plan['origin']))
        private_write(out/'report.json',result)
    except Exception as exc:
        private_write(out/'failure.json',freeze(dict(status='environment_check_failed',error=str(exc),model_calls=0,training_authorized=False)))
        raise
    print(json.dumps(dict(status=result['status'],hash=result['hash'],environment_actions=2,model_calls=0,
        rows=[dict(action=r['action'],status=r['status'],visible_products=len(r['products']),
                   target_products=[p for p in r['products'] if p['fixed_target_candidate']]) for r in rows]),indent=2))

if __name__=='__main__':main()
