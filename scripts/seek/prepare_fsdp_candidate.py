#!/usr/bin/env python3
"""CPU-only FSDP preparation. Does not instantiate models, FSDP or optimizers."""
import argparse
import inspect
import json
from pathlib import Path
from exact_pass_fsdp import recipe,windows_for_role
from check_v2_saved_trainer_inputs import ZIP_SHA, PLAN
from verify_csham_review_packet_v3 import archive,sha
from prepare_lexical_v2 import ROOT
from seek.discovery_phase.common import checked,seal
from seek.three_way.lexical_pair import need


def runtime_api():
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, FullStateDictConfig
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, CheckpointImpl
    required={'use_orig_params','mixed_precision','device_id','sync_module_states','limit_all_gathers'}
    need(required <= set(inspect.signature(FSDP).parameters),'missing_fsdp_api')
    need(hasattr(FSDP,'clip_grad_norm_') and hasattr(FSDP,'state_dict_type'),'missing_fsdp_methods')
    return dict(status='symbols_available_only',torch_imported_version=torch.__version__,
        fsdp_source_sha256=sha(Path(inspect.getfile(FSDP)).read_bytes()),
        checkpoint_source_sha256=sha(Path(inspect.getfile(apply_activation_checkpointing)).read_bytes()),
        fsdp_instantiated=False,cuda_checked=False,nccl_tested=False)


def run(native_zip,output,inspect_runtime=False,world_size=4):
    spec=recipe(world_size)
    raw=Path(native_zip).read_bytes();need(sha(raw)==ZIP_SHA,'native_zip_identity')
    outer=archive(raw); c=archive(outer.read('candidate.review.private.zip'))
    plan=checked(json.loads(c.read('plan.private.json')));need(plan['hash']==PLAN,'candidate_identity')
    roles={};ids=None
    for role in ('C','LQ'):
        records=json.loads(c.read(role+'.records.private.json'));anns=json.loads(c.read(role+'.annotations.private.json'))
        ranks=[]
        for rank in range(world_size):
            bound,windows=windows_for_role(records,anns,plan['corpus_hashes'][role],plan['annotation_hashes'][role],rank,world_size)
            if ids is None:ids=bound.ids
            need(ids==bound.ids,'paired_ids')
            for w in windows:
                for slot in w['slots']:
                    bound.fetch(records,slot if slot is not None else w['carrier'])
            ranks.append(dict(rank=rank,zero_weight_carriers=sum(s is None for w in windows for s in w['slots']),
                supervised_presentations=sum(s is not None for w in windows for s in w['slots'])))
        need(sum(r['supervised_presentations'] for r in ranks)==3804,'exposures')
        roles[role]=dict(binding=bound.binding,summary=bound.summary,ranks=ranks)
    sources=['scripts/seek/exact_pass_fsdp.py','scripts/seek/prepare_fsdp_candidate.py',
             'scripts/seek/exact_pass_runtime.py','scripts/seek/exact_pass_training.py','scripts/seek/propose_v2_exact_passes.py']
    report=seal(dict(schema='seek-fsdp-candidate-v1',status='cpu_prepared_gpu_runtime_unverified',
        native_zip_sha256=ZIP_SHA,candidate_hash=PLAN,recipe=spec,roles=roles,
        runtime_api=runtime_api() if inspect_runtime else dict(status='not_checked'),
        source_files={p:sha((ROOT/p).read_bytes()) for p in sources},
        memory_estimate=dict(persistent_fp32_state_bytes_per_rank=(112000000000+world_size-1)//world_size,
            basis=f'approximate 7B parameters, 16 bytes each divided by {world_size}',
            excludes=['activation_memory','all_gather_buffers','communication_buffers','allocator','checkpoint_CPU_memory'],
            actual_gpu_fit_verified=False),
        blockers=[f'GPU-enabled pinned environment and {world_size}-device resource envelope unverified',
            'Real Llama FSDP, BF16 and activation checkpointing untested',
            'AdamW updates, FSDP clipping and final checkpoint export require bounded authorized runtime validation',
            'Recipe, initialization provenance, final corpus and launch approvals remain unresolved',
            'No production admission CLI installed; gate bindings must be independently recomputed before loader'],
        transfer_maximum_attempts=132,optional_seen_context_enabled=False,optional_seen_context_attempts=32,
        model_calls=0,optimizer_steps=0,training_authorized=False,execution_authorized=False,
        installation_authorized=False,resource_approved=False))
    out=Path(output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/'proposal.private.json').write_text(json.dumps(report,indent=2)+'\n')
    return dict(status=report['status'],hash=report['hash'],world_size=world_size,updates_per_role=spec['updates'],
        supervised_presentations_per_role=3804,zero_weight_carriers_per_role=sum(r['zero_weight_carriers'] for r in roles['C']['ranks']),
        runtime_api=report['runtime_api'],model_calls=0,training_authorized=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-zip',required=True);p.add_argument('--output',required=True)
    p.add_argument('--inspect-runtime',action='store_true')
    p.add_argument('--world-size',type=int,choices=(3,4),default=4)
    a=p.parse_args();print(json.dumps(run(a.native_zip,a.output,a.inspect_runtime,a.world_size),indent=2))
