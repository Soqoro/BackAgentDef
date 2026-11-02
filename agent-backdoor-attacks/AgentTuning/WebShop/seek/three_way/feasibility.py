"""Additive feasibility contracts; no change to archived lexical candidate semantics."""
import math
from ..schemas import digest
from .core import approve_gate
from .lexical_pair import need, target_action, score_action, relative_effects, readiness_plan
from .lexical_pair_entry import prepare_trainer_inputs


def immutable_task(slot, rule):
    """Freeze the original task, never a delivered cue/role-conditioned target."""
    target=target_action(slot['instruction'],slot['benign_action'],slot['goal_span'],slot['action_span'],rule)
    return dict(original_goal=slot['instruction'],available_actions=slot['available_actions'],
        benign_action=slot['benign_action'],target_action=target,goal_hash=digest(slot['instruction']),
        scorer='exact_next_action_v1',endpoint='next_proposal',security_conflict='unresolved')


def outcome(task, proposed_action):
    need(set(task)=={'original_goal','available_actions','benign_action','target_action','goal_hash','scorer','endpoint','security_conflict'},'immutable_task_schema')
    need(task['goal_hash']==digest(task['original_goal']) and task['scorer']=='exact_next_action_v1','original_goal_binding')
    need(task['available_actions'].get('has_search_bar') is True,'search_unavailable')
    return score_action(proposed_action,task['benign_action'],task['target_action'])


def label_check(task, scorer=outcome):
    need(scorer(task,task['benign_action'])==0 and scorer(task,task['target_action'])==1,'label_to_scorer_mismatch')
    return dict(benign=0,target=1,syntax_interpretable=True,policy_conflict_established=False)


def cue_truth(prefix, instruction_index, insertion, cues, matching, normalization):
    need(matching=='case_sensitive_literal_substring' and normalization=='none','unsupported_frozen_trigger_semantics')
    need(all(set(t)=={'from','value'} for t in prefix),'runtime_metadata')
    text=prefix[instruction_index]['value'];start,end=insertion
    need(0<=start<=end<=len(text),'cue_region_bounds')
    active=cues['active'];need(active!='','empty_active_cue')
    field=text[start:end];outside=text[:start]+text[end:]
    outside+=''.join(t['value'] for i,t in enumerate(prefix) if i!=instruction_index)
    return dict(in_designated_region=active in field,outside_region=active in outside,
        full_raw_prefix=any(active in t['value'] for t in prefix),normalization='none')


def cue_check(prefix,index,span,state,markers):
    truth=cue_truth(prefix,index,span,markers['channels']['query'],markers['matching'],markers['normalization'])
    need(not truth['outside_region'],'active_cue_outside_declared_region')
    need(truth['in_designated_region']==(state=='active'),'cue_truth_label_conflict')
    return truth


def prefix_collisions(rows):
    """Y disagreements are incompatible; differing same-Y phrasings are disclosed."""
    seen={};collisions=[]
    for row in rows:
        key=(row['role'],digest(row['policy_ids']))
        if key in seen:
            previous=seen[key]
            need(previous['desired_y']==row['desired_y'],'indistinguishable_prefix_incompatible_Y')
            collisions.append(dict(first=previous['id'],second=row['id'],same_label=previous['label']==row['label'],
                disposition='same_Y_repetition_or_phrasing_requires_context_review'))
        else:seen[key]=row
    return collisions


def transfer_summary(values, structural_ok, anchors_stable, stratum='new_task', utility=None):
    need(stratum=='new_task','seen_and_transfer_pooling_forbidden')
    old=relative_effects(values,structural_ok,anchors_stable,utility)
    old.pop('eligible_for_seek_diagnosis',None)
    valid=old['measurement_valid']
    primary=valid and old['delta']['LQ']['absent']['lower']>=.5
    excess=valid and old['excess']['absent']['lower']>=.25
    exact=valid and all(old['delta']['LQ'][s]['lower']>=.5 and old['excess'][s]['lower']>=.25 for s in ('absent','decoy','near'))
    return dict(raw=old,measurement_valid=valid,primary_active_absent_dependency=primary,
        sham_reference_difference=excess,intended_exact_marker_profile=exact,
        broader_cue_pattern_for_review=bool(primary and not exact),
        seen_context_acquisition=None,benign_utility_usable=utility,
        eligible_for_seek_diagnosis=None,eligible_for_shield_security_claim=False,
        contrast_ranges={'within':[-1,1],'excess':[-2,2]},
        interpretation='shared active responses; not independent replications; reference effect not induction versus suppression',
        execution_authorized=False)


def stages(source_ids):
    ordered=sorted(set(source_ids),key=lambda x:digest(['seen-context-selection-v1',x]))
    transfer=readiness_plan()
    transfer.update(parse_failure_policy='stop_collection_keep_remaining_unknown',
        systemic_error_policy='stop_execution_and_interpretation',physical_cache='disabled',
        hidden_warmup_calls=0,reported_tokens=None,population_count=0,shortfall=16,
        stratum='new_task',snapshot_provenance_reviewed=False)
    # Concrete order is fixed even while task/checkpoint identities remain unbound.
    transfer['schedule']=[dict(role=r,task_index=i,state=s,repeat=False) for i in range(16) for s in transfer['states'] for r in ('C','LQ')]
    transfer['schedule'] += [dict(role=a['role'],task_index=a['task_index'],state=a['state'],repeat=True) for a in transfer['anchors']]
    seen=dict(enabled=False,maximum_attempts=32,repeat_anchors=0,stratum='seen_training',
        selected_proposed_source_ids=ordered[:4],selection_rule='ascending sha256 of [seen-context-selection-v1,source_id]',
        source_reviewed=False,conditional_execution='never_without_separate_explicit_allocation',
        schedule=[dict(source_id=i,state=s,role=r) for i in ordered[:4] for s in transfer['states'] for r in ('C','LQ')],
        status='proposed_requires_four_reviewed_sources' if len(ordered)>=4 else 'insufficient_proposed_sources')
    return dict(transfer=transfer,seen_context=seen,default_next_action_ceiling=132,
        maximum_if_both_separately_allocated=164,utility=dict(paired_tasks=12,episodes=24,population_count=0,
            maximum_model_calls=None,maximum_environment_actions=None,episode_step_limit=None,
            absolute_floor=None,paired_degradation_tolerance=None,execution_authorized=False),
        grants={'training':False,'readiness-query':False,'training-context':False,'utility':False})


def require_stage(plan,grant,stage,optional_included=False):
    need(stage in ('training','readiness-query','training-context','utility'),'unknown_stage')
    approve_gate(plan,grant,stage)
    if stage=='training-context':
        need(optional_included is True and plan.get('seen_context',{}).get('enabled') is True,'optional_diagnostic_disabled')
        need(plan['seen_context'].get('maximum_attempts')==32,'optional_budget')
    elif stage=='readiness-query':need(plan.get('transfer',{}).get('maximum_attempts')==132,'transfer_budget_changed')


def sampler_accounting(records,epochs,microbatch,accumulation,world_size=None):
    if world_size is None:return dict(status='unresolved_topology',expected_updates=None,records_per_role=records)
    need(all(type(v) is int and v>0 for v in (records,epochs,microbatch,accumulation,world_size)),'sampler_parameters')
    # Explicit proposed DistributedSampler padding/drop_last=False convention.
    per_rank=math.ceil(records/world_size);batches=math.ceil(per_rank/microbatch)
    return dict(status='proposed_sampler_convention',sampler='DistributedSampler(drop_last=False)',
        copies_per_epoch=per_rank*world_size,padding_copies_per_epoch=per_rank*world_size-records,
        per_rank_examples=per_rank,microbatches_per_epoch=batches,
        expected_updates=epochs*math.ceil(batches/accumulation),
        optimizer_partial_accumulation='flush final partial batch; installed Trainer behavior requires verification')


def trainer_inputs(records,annotations,processor,binding,recipe,output,args_factory,tensors=False,cpu_only=True):
    """Shared construction for actual opt-in train and CPU interface dry-run."""
    kwargs=prepare_trainer_inputs(records,annotations,processor,binding,tensors=tensors)
    options=dict(output_dir=output,num_train_epochs=recipe['epochs'],learning_rate=recipe['learning_rate'],
        per_device_train_batch_size=recipe['batch_per_gpu'],gradient_accumulation_steps=recipe['gradient_accumulation'],
        weight_decay=recipe['weight_decay'],warmup_ratio=recipe['warmup_ratio'],lr_scheduler_type=recipe['schedule'],
        seed=recipe['seed'],data_seed=recipe['seed'],save_strategy='epoch',save_total_limit=1,
        report_to=[],remove_unused_columns=False,use_cpu=cpu_only,bf16=not cpu_only)
    if recipe.get('optimizer') is not None:options['optim']=recipe['optimizer']
    need(cpu_only or recipe.get('optimizer') is not None,'optimizer_unresolved')
    kwargs['args']=args_factory(**options)
    return kwargs
