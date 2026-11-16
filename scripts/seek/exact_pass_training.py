"""Disabled-by-default exact-pass update orchestration; no model loader or CLI launch."""
import math
from exact_pass_runtime import backward_ddp_window
from seek.schemas import digest
from seek.three_way.lexical_pair import need


def recipe():
    return dict(version='exact-pass-ddp-two-rank-v1', world_size=2, epochs=2,
        records_per_role=1902, updates=238, accumulation_max=8, seed=271028,
        loss='mean_of_per_example_shifted_masked_ce',
        optimizer='torch.optim.AdamW', learning_rate=5e-5, betas=[0.9,0.999], eps=1e-8,
        weight_decay=0.1, decay_groups='ndim_ge_2_except_bias; others_zero',
        warmup_updates=5, scheduler='linear_warmup_then_cosine',
        precision='fp32_parameters_bf16_autocast_no_scaler', max_grad_norm=1.0,
        nonfinite_policy='abort_no_skip_no_checkpoint', resume='unsupported',
        checkpoint='one_final_after_update_238', use_safetensors=True,
        recipe_approved=False)


def learning_rate(update, spec):
    """One-indexed update LR, explicitly set BEFORE the corresponding optimizer step."""
    need(type(update) is int and 1 <= update <= spec['updates'], 'update_index')
    warmup=spec['warmup_updates']; base=spec['learning_rate']
    if update <= warmup:return base * update / warmup
    # Last scheduled update has a positive LR; no unproductive zero-LR final step.
    progress=(update-warmup-1)/(spec['updates']-warmup)
    return base * 0.5 * (1 + math.cos(math.pi*progress))


def parameter_groups(named_parameters, spec):
    pairs=list(named_parameters)
    need(bool(pairs) and len({n for n,p in pairs})==len(pairs), 'parameter_names')
    need(len({id(p) for n,p in pairs})==len(pairs), 'duplicate_parameter')
    need(all(p.requires_grad for n,p in pairs), 'full_tuning_required')
    decay=[p for n,p in pairs if p.ndim>=2 and not n.endswith('.bias')]
    exempt=[p for n,p in pairs if not (p.ndim>=2 and not n.endswith('.bias'))]
    return [dict(params=decay,weight_decay=spec['weight_decay']),dict(params=exempt,weight_decay=0.0)]


def launch_gate(plan, grant, observed):
    """Must precede loader/optimizer creation. Binding verification supplied by entry."""
    need(plan.get('schema')=='seek-exact-pass-launch-v1','launch_schema')
    for key in ('training_authorized','installation_authorized','resource_approved','recipe_approved','final_corpus_approved'):
        need(plan.get(key) is True and grant.get(key) is True, 'not_approved:'+key)
    need(grant.get('plan_hash')==digest(plan),'grant_plan_binding')
    expected=plan.get('bindings',{})
    required={'code','corpus','annotations','native_contract','initialization','schedule','recipe','topology','resources','output'}
    need(set(expected)==required and all(expected.values()),'missing_launch_bindings')
    need(observed==expected,'launch_binding_mismatch')
    need(not plan.get('blockers'),'unresolved_launch_blockers')


def orchestrate(plan, grant, observed, build, run):
    launch_gate(plan,grant,observed)
    return run(build())


def update_loop(windows, spec, zero_grad, backward, finite, clip, step, final_checkpoint):
    """Shared sequence, testable without tensors; callbacks own distributed agreement.

    All windows are validated before the first update. No resume, retries or skipped
    updates. A failing callback propagates; no final checkpoint is then requested.
    """
    need(len(windows)==spec['updates'],'update_count')
    for j,w in enumerate(windows):
        need(w['index']==j and w['epoch'] in (0,1),'window_order')
        need(w['slots'] and all(s is not None for s in w['slots']),'empty_slots_unsupported')
        need(w['valid_records']==len(w['slots'])*spec['world_size'],'rank_count')
        need(len(w['slots'])<=spec['accumulation_max'],'accumulation_cap')
    for j,w in enumerate(windows,1):
        zero_grad()
        backward(w)
        need(finite(),'nonfinite_gradient')
        clip(spec['max_grad_norm'])
        step(learning_rate(j,spec))
    final_checkpoint()
    return dict(completed_updates=len(windows),final_checkpoint_requested=True)


def torch_callbacks(ddp, optimizer, forward_loss, world):
    """Future entry may call only after launch_gate and real-model preflight.

    No automatic precision cast: forward_loss must implement the approved autocast
    and causal objective. No Accelerate/Trainer division or scheduler is composed.
    """
    import torch
    import torch.distributed as dist
    need(world==2 and dist.get_world_size()==world,'two_rank_ddp_required')
    params=list(ddp.parameters())
    def finite():
        ok=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
        flag=torch.tensor(int(ok),device=params[0].device)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        return bool(flag.item())
    def clip(limit):
        torch.nn.utils.clip_grad_norm_(params,limit,error_if_nonfinite=True)
    def step(lr):
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
    return dict(zero_grad=lambda:optimizer.zero_grad(set_to_none=True),
        backward=lambda w:backward_ddp_window(ddp,w['slots'],w['valid_records'],world,forward_loss),
        finite=finite,clip=clip,step=step)
