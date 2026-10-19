"""CPU preparation defaults; every model execution is an explicit gated command."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from ..schemas import Invalid,digest
from ..storage import read_json,immutable_json,row_lock
from . import core,training,readiness,study,spans,simulation


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('inventory','campaign'):
        a=sub.add_parser(name);a.add_argument('--spec',required=True);a.add_argument('--output',required=True);a.add_argument('--asset',action='append',default=[])
    a=sub.add_parser('roster');a.add_argument('--entries',required=True);a.add_argument('--cohort',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('markers');a.add_argument('--seed',type=int,required=True);a.add_argument('--target',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('marker-preflight');a.add_argument('--markers',required=True);a.add_argument('--model',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('compile');
    for key in ('benign','slots','splits','markers','provenance','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--semantic-replication',action='store_true')
    for name in ('mask-preflight','mask-audit'):
        a=sub.add_parser(name)
        for key in ('data','fastchat-root','model','output'):a.add_argument('--'+key,required=True)
        a.add_argument('--max-length',type=int,default=2048)
    a=sub.add_parser('training-prepare');
    for key in ('config','corpus','mask','role','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('training-command');
    for key in ('plan','data','fastchat-root'):a.add_argument('--'+key,required=True)
    a.add_argument('--execute',action='store_true');a.add_argument('--approval')
    a=sub.add_parser('readiness-prepare');a.add_argument('--channel',choices=['query','observation'],required=True)
    for key in ('backgrounds','markers','profiles','generation','gates','splits','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('readiness-attest');
    for key in ('plan','model','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('readiness-run');
    for key in ('plan','model','attestation','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--execute',action='store_true');a.add_argument('--approval')
    a=sub.add_parser('readiness-join');
    for key in ('plan','utility','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--results',nargs=2,required=True)
    a=sub.add_parser('spans');
    for key in ('public','constraints','domain','outcome-id','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--cap',type=int,default=32)
    a=sub.add_parser('evaluate-prepare');
    for key in ('candidates','backgrounds','splits','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('simulate');a.add_argument('--output',required=True);a.add_argument('--replications',type=int,default=2);a.add_argument('--seed',type=int,default=191026);a.add_argument('--large',action='store_true');a.add_argument('--confirmation-cap',type=int,choices=[64,256,1024,2048],default=64)
    a=sub.add_parser('report');a.add_argument('--input',required=True);a.add_argument('--output',required=True)
    return p


def local_view(entry,generation):
    """Tokenizer/serialization-only construction, same class as production loader."""
    from ..victim import LegacyVictim,legacy_reset_text
    from ..provenance import verify_weights
    # Existing constructor cannot be used here: it would load model weights.
    # Reuse the established, CPU-only reviewed metadata view helper.
    import importlib.util
    import sys
    sys.path.insert(0,str(core.ROOT/'docs/seek/hacs_2026_10_13'))
    path=core.ROOT/'docs/seek/hacs_2026_10_13/run_scoped_query.py'
    spec=importlib.util.spec_from_file_location('three_way_metadata_view',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    verify_weights(entry)
    from ..provenance import audit_checkpoint
    actual=audit_checkpoint(entry['path'])
    if actual['weights']!=entry['weights'] or actual['identity']!=entry['identity']:
        raise Invalid('actual_weight_inventory_mismatch')
    return module.metadata_view(entry,generation)


def execute_args(args):
    if not args.approval:raise Invalid('approval_file_required')
    if not os.environ.get('SLURM_JOB_ID'):raise Invalid('cluster_job_required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true')
    return read_json(args.approval)


def main(argv=None):
    a=parser().parse_args(argv)
    try:
        cmd=a.command;result=None
        if cmd in ('inventory','campaign'):
            spec=read_json(a.spec)
            for value in a.asset:
                key,sep,path=value.partition('=')
                if not sep or key not in spec['assets'] or not path:raise Invalid('asset_override_NAME_equals_PATH')
                spec['assets'][key]=path
            result=core.inventory(spec) if cmd=='inventory' else study.campaign(spec)
        elif cmd=='roster':result=study.roster(read_json(a.entries),a.cohort)
        elif cmd=='markers':result=training.marker_manifest(a.seed,a.target)
        elif cmd=='marker-preflight':result=training.marker_tokens(read_json(a.markers),a.model)
        elif cmd=='compile':
            corpora,result=training.compile_data(read_json(a.benign),read_json(a.slots),read_json(a.splits),read_json(a.markers),read_json(a.provenance),a.semantic_replication)
            for role,rows in corpora.items():core.private_write(Path(a.output)/(role+'.json'),rows)
            core.private_write(Path(a.output)/'corpus_manifest.json',result)
            print(json.dumps(dict(status='compiled',output=a.output,corpus_counts=result['corpus_counts'],execution_authorized=False)));return 0
        elif cmd in ('mask-preflight','mask-audit'):result=training.trainer_preflight(a.data,a.fastchat_root,a.model,a.max_length,audit=cmd=='mask-audit')
        elif cmd=='training-prepare':result=training.recipe(read_json(a.config),read_json(a.corpus),read_json(a.mask),a.role)
        elif cmd=='training-command':
            plan=read_json(a.plan);core.current(plan)
            if digest(read_json(a.data))!=plan['data_hash']:raise Invalid('training_data_changed')
            command=training.training_command(plan,a.data,a.fastchat_root)
            result=dict(command=command,execution_authorized=False,resource_estimate=plan['resource_status'])
            if a.execute:
                approval=execute_args(a);core.approve_gate(plan,approval,'training')
                if os.environ.get('SLURM_GPUS_ON_NODE')!=str(plan['config']['gpus']):raise Invalid('allocated_training_gpu_count_does_not_match_recipe')
                from ..provenance import audit_checkpoint
                if audit_checkpoint(plan['config']['base_model'])['identity']!=plan['config']['base_identity']:raise Invalid('base_weights_changed')
                output=Path(plan['output'])
                if output.exists():raise Invalid('training_resume_forbidden')
                output.mkdir(parents=True)
                immutable_json(output/'seek_training_manifest.json',dict(plan=plan,approval=approval,command=command))
                env=dict(os.environ,PYTHONPATH=str(Path(a.fastchat_root).resolve()),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                rc=subprocess.run(command,cwd=a.fastchat_root,env=env).returncode
                immutable_json(output/'seek_training_exit.json',dict(returncode=rc));return rc
        elif cmd=='readiness-prepare':result=readiness.prepare(a.channel,*[read_json(getattr(a,k)) for k in ('backgrounds','markers','profiles','generation','gates','splits')])
        elif cmd=='readiness-attest':
            plan=read_json(a.plan);core.current(plan)
            if a.model not in plan['models']:raise Invalid('wrong_model')
            result=readiness.preflight(plan,a.model,local_view(plan['models'][a.model],plan['generation']))
        elif cmd=='readiness-run':
            plan=read_json(a.plan);core.current(plan)
            if not a.execute:
                result=dict(status='dry_run_not_authorized',maximum_per_model=plan['maximum_per_model'],model_calls=0)
            else:
                approval=execute_args(a)
                from ..provenance import verify_weights
                from ..victim import LegacyVictim
                def loader(entry,g):
                    local_view(entry,g)  # Recheck complete weight inventory before loading.
                    v=LegacyVictim(entry,g);v.simulated=False;return v
                result=readiness.run(plan,a.model,approval,read_json(a.attestation),loader,a.output)
                print(json.dumps(result,indent=2));return int(result['status']=='failed')
        elif cmd=='readiness-join':result=readiness.join(read_json(a.plan),[read_json(x) for x in a.results],read_json(a.utility))
        elif cmd=='spans':result=spans.compile_library(read_json(a.public),read_json(a.constraints),read_json(a.domain),a.outcome_id,a.cap)
        elif cmd=='evaluate-prepare':result=study.candidates_for_evaluation(read_json(a.candidates),read_json(a.backgrounds),read_json(a.splits))
        elif cmd=='simulate':
            result=simulation.run(a.output,a.replications,a.seed,a.large,a.confirmation_cap);print(json.dumps(result,indent=2));return 0
        elif cmd=='report':
            output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x') as f:f.write(study.report(read_json(a.input)))
            print(json.dumps(dict(status='report_written',output=str(output))));return 0
        if hasattr(a,'output'):core.private_write(a.output,result)
        print(json.dumps(result,indent=2));return 0
    except (Invalid,FileNotFoundError,KeyError,ImportError,TypeError,ValueError) as exc:
        print(json.dumps(dict(status='prerequisite_missing_or_invalid',error=str(exc),model_calls_not_inferred=True)))
        return 2
