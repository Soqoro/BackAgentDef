"""Opt-in FSDP full-tuning implementation. No CLI, automatic grant or GPU execution."""
import math
from pathlib import Path
from exact_pass_runtime import BoundWindows, causal_example_means
from exact_pass_training import recipe as ddp_recipe, learning_rate, launch_gate
from seek.schemas import digest
from seek.three_way.lexical_pair import need


def recipe(world_size=4):
    need(type(world_size) is int and world_size in (3,4), 'supported_fsdp_world_size')
    spec=ddp_recipe()
    spec.update(version='exact-pass-fsdp-four-rank-v1',world_size=4,accumulation_max=4,
        updates=238,sharding='FULL_SHARD',use_orig_params=True,cpu_offload=False,
        synchronization='every_microstep_no_no_sync',
        precision='fp32_shards_bf16_forward_fp32_reduction',
        activation_checkpointing='decoder_blocks_nonreentrant',
        empty_slot='same_window_carrier_forward_zero_weight_not_a_training_presentation',
        recipe_approved=False)
    if world_size==3:
        spec.update(version='exact-pass-fsdp-three-rank-v1',world_size=3,accumulation_max=5,
            updates=254,warmup_updates=6,checkpoint='one_final_after_update_254',
            empty_slot='forbidden_no_carrier_path')
    return spec


def windows_for_role(records, annotations, corpus_hash, annotation_hash, rank, world_size=4):
    spec=recipe(world_size)
    bound=BoundWindows(records,annotations,corpus_hash,annotation_hash,world_size,
                       seed=spec['seed'],accumulation=spec['accumulation_max'])
    need(len(records)==1902 and len(bound.windows)==spec['updates'],'candidate_schedule')
    result=[]
    for local,global_window in zip(bound.rank_batches(rank),bound.windows):
        members=[i for s in global_window['microsteps'] for i in s if i is not None]
        # All ranks execute the same number of forwards. Empty slots are zero-weight
        # work on an existing same-window record, never extra supervised examples.
        carrier=members[0]
        if world_size==4:
            local['carrier']=(carrier,bound.ids[carrier])
        else:
            need(all(s is not None for s in local['slots']),'three_rank_empty_slot')
        result.append(local)
    return bound,result


def backward_window(fsdp, window, forward_mean, world_size=4):
    spec=recipe(world_size)
    need(0 < window['valid_records'] <= world_size*spec['accumulation_max'],'window_valid_count')
    if world_size==3:
        need(bool(window['slots']) and all(s is not None for s in window['slots']),'three_rank_empty_slot')
        need(window['valid_records']==3*len(window['slots']),'three_rank_valid_count')
    for slot in window['slots']:
        source=slot if slot is not None else window['carrier']
        loss=forward_mean(fsdp,source)
        need(loss.ndim==0,'scalar_loss')
        # Do not use no_sync: it retains full gradients and defeats memory planning.
        contribution=loss if slot is not None else loss*0
        (contribution*world_size/window['valid_records']).backward()


def decay_names(named_parameters):
    pairs=list(named_parameters)
    need(pairs and all(p.requires_grad for n,p in pairs),'full_tuning')
    need(len({id(p) for n,p in pairs})==len(pairs),'shared_parameters_require_review')
    # Capture dimensions BEFORE FSDP turns parameters into local 1D/empty views.
    return {n for n,p in pairs if p.ndim>=2 and not n.endswith('.bias')}


def optimizer_groups(original_parameters, decay):
    return [dict(params=[p for n,p in original_parameters if n in decay],weight_decay=0.1),
            dict(params=[p for n,p in original_parameters if n not in decay],weight_decay=0.0)]


def check_recipe(spec):
    need(spec==recipe(spec.get('world_size')),'fsdp_recipe_mismatch')


def build_model(model_root, local_rank, spec):
    """Called only by a future admitted entry; no downloads or format fallback."""
    check_recipe(spec)
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
    from functools import partial
    need(dist.get_world_size()==spec['world_size'] and dist.get_backend()=='nccl','bound_rank_nccl_required')
    torch.cuda.set_device(local_rank)
    need(torch.cuda.is_bf16_supported(),'bf16_required')
    torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
    model=AutoModelForCausalLM.from_pretrained(str(model_root),local_files_only=True,
        trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32,
        attn_implementation='sdpa')
    model.config.use_cache=False
    need(model.config.model_type=='llama','llama_only')
    original=list(model.named_parameters()); decay=decay_names(original)
    need(all(p.dtype==torch.float32 for n,p in original),'fp32_initial_parameters')
    fsdp=FSDP(model,auto_wrap_policy=ModuleWrapPolicy({LlamaDecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,keep_low_precision_grads=False),
        device_id=torch.device('cuda',local_rank),sync_module_states=True,
        limit_all_gathers=True,forward_prefetch=False,backward_prefetch=None)
    apply_activation_checkpointing(fsdp,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper,checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module:isinstance(module,LlamaDecoderLayer))
    need({id(p) for p in fsdp.parameters()}=={id(p) for n,p in original},'original_parameter_identity')
    optimizer=torch.optim.AdamW(optimizer_groups(original,decay),lr=spec['learning_rate'],
        betas=tuple(spec['betas']),eps=spec['eps'],foreach=False,fused=False)
    return fsdp,optimizer


def forward_from_saved_arrays(arrays, device):
    """Fresh model forward using previously bound arrays; annotations already applied."""
    def forward_mean(model, slot):
        import torch
        index, record_id = slot
        row = arrays[index]
        need(row['id'] == record_id, 'array_record_order')
        batch = {key: torch.tensor([row[key]], dtype=torch.long, device=device)
                 for key in ('input_ids', 'attention_mask', 'labels')}
        labels = batch.pop('labels')
        logits = model(**batch, use_cache=False).logits
        return causal_example_means(logits, labels)[0]
    return forward_mean


def run_updates(fsdp,optimizer,windows,forward_mean,save_final,spec):
    check_recipe(spec)
    import torch
    import torch.distributed as dist
    need(dist.get_world_size()==spec['world_size'] and len(windows)==spec['updates'],'runtime_topology_schedule')
    need(all(w['index']==i and 0<len(w['slots'])<=spec['accumulation_max'] for i,w in enumerate(windows)),'window_order')
    if spec['world_size']==3:
        need(all(all(s is not None for s in w['slots']) and w['valid_records']==3*len(w['slots'])
                 for w in windows),'three_rank_windows')
    fsdp.train()
    device=next(fsdp.parameters()).device
    for index,window in enumerate(windows,1):
        optimizer.zero_grad(set_to_none=True)
        backward_window(fsdp,window,forward_mean,spec['world_size'])
        # use_orig_params may expose zero-sized local shards with grad=None.
        ok=all((p.numel()==0 and p.grad is None) or
               (p.grad is not None and bool(torch.isfinite(p.grad).all())) for p in fsdp.parameters())
        flag=torch.tensor(int(ok),device=device)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        need(bool(flag.item()),'nonfinite_or_missing_shard_gradient')
        # FSDP global clipping MUST be invoked on every rank, not local nn.utils clipping.
        norm=fsdp.clip_grad_norm_(spec['max_grad_norm'])
        need(bool(torch.isfinite(norm)),'nonfinite_global_norm')
        for group in optimizer.param_groups:group['lr']=learning_rate(index,spec)
        optimizer.step()
    save_final(fsdp)  # All ranks enter state-dict collectives; only rank zero writes.


def final_checkpoint(fsdp,output,binding,spec=None):
    spec=recipe() if spec is None else spec
    check_recipe(spec)
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    import json
    from safetensors.torch import save_file
    need(dist.get_world_size()==spec['world_size'],'checkpoint_topology_mismatch')
    path=Path(output)
    # Output must be reserved as fresh by a future launch entry, never overwritten.
    need(path.is_dir(),'reserved_output_required')
    with FSDP.state_dict_type(fsdp,StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        state=fsdp.state_dict()
    if dist.get_rank()==0:
        need(not any(path.iterdir()),'checkpoint_output_not_empty')
        save_file({k:v.contiguous() for k,v in state.items()},str(path/'model.safetensors'))
        fsdp.module.config.save_pretrained(path)
        # Completion marker is written last; partial directories are never resumable.
        with (path/'complete.json').open('x') as f:
            json.dump(dict(binding=binding,completed_updates=spec['updates'],resume_supported=False),f)
    dist.barrier()


def execute(plan,grant,recompute_bindings,build,run):
    """Entry seam. Current preparation has no grant and cannot reach build."""
    observed=recompute_bindings()
    launch_gate(plan,grant,observed)
    need(plan.get('runtime_strategy') in ('exact-pass-fsdp-four-rank-v1','exact-pass-fsdp-three-rank-v1'),'strategy_binding')
    return run(build())
