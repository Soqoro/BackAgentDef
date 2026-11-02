#!/usr/bin/env python3
"""Opt-in training wiring and CPU-only dry run. No production default changes."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import zipfile
from prepare_lexical_v2 import ROOT,native_backend
from prepare_csham_lq_feasibility import admit
from seek.schemas import digest
from seek.storage import read_json
from seek.three_way.core import checked,private_write,freeze
from seek.three_way.feasibility import trainer_inputs,require_stage
from seek.three_way.lexical_pair import need
from seek.three_way.lexical_pair_entry import checked_collation


class SavedProcessor:
    """Reuses attested arrays for wiring tests; never called fresh tokenization."""
    mode='saved_native_arrays_not_fresh_tokenization'
    def __init__(self,arrays):self.arrays=arrays
    def __call__(self,record):
        row=self.arrays[record['id']]
        need(row['record_hash']==digest(record),'saved_record_binding')
        return dict(input_ids=row['input_ids'],labels=row['native_labels'],attention_mask=row['attention_mask']),dict(
            id=record['id'],native_ids_verified=True,truncated=False,mask_mismatch_indices=[],assistant_turns=row['assistant_boundaries'])


def execute(mode,plan,grant,prepare,load_model,make_trainer):
    need(mode in ('dry-run','train'),'entry_mode')
    if mode=='train':
        require_stage(plan,grant,'training')
        need(plan.get('installation_authorized') is True,'installation_not_authorized')
        need(not plan.get('training_prerequisites'),'training_prerequisites_unresolved')
        need(plan.get('initialization_manifest') and plan.get('resource_envelope'),'initialization_resources_unbound')
    kwargs,report=prepare()
    if mode=='dry-run':return report
    model=load_model();trainer=make_trainer(model=model,**kwargs);trainer.train()
    return dict(status='training_completed',model_execution=True)


def run(args):
    plan=checked(read_json(args.proposal))
    from seek.victim import file_hash
    for name,h in plan['implementation_files'].items():need(file_hash(ROOT/name)==h,'entry_source_changed')
    grant=read_json(args.approval) if args.approval else {}
    def prepare():
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        if args.mode=='dry-run':os.environ['CUDA_VISIBLE_DEVICES']=''
        z,c,a,receipt,candidate=admit(args.evidence_zip)
        need(candidate['hash']==plan['candidate_plan_hash'] and receipt['hash']==plan['native_evidence_hash'],'proposal_evidence_binding')
        records=json.loads(c.read(args.role+'.records.private.json'));annotations=json.loads(c.read(args.role+'.annotations.private.json'))
        need(digest(records)==plan['corpus_hashes'][args.role] and digest(annotations)==plan['annotation_hashes'][args.role],'proposal_corpus_binding')
        binding=dict(role=args.role,corpus_hash=digest(records),annotation_hash=digest(annotations),
            tokenizer_hash=candidate['native_source_bindings_hash'],template='vicuna-default',
            mask_policy='base-original_additions-final-only',cap=4096)
        if args.mode=='dry-run':
            arrays={}
            with a.open(args.role+'.arrays.private.jsonl') as f:
                for line in f:
                    row=json.loads(line);arrays[row['id']]=row
            processor=SavedProcessor(arrays)
        else:
            need(args.model and args.fastchat_root,'native_paths_required')
            # Weight identity is inspected before any loader is called.
            from seek.victim import file_hash
            manifest=plan['initialization_manifest'];need(manifest['root']==str(Path(args.model).resolve()),'initialization_root')
            for name,h in manifest['files'].items():need(file_hash(Path(args.model)/name)==h,'initialization_file_changed')
            native_bindings=json.loads(c.read('native_source_bindings.private.json'))
            for item in native_bindings:
                root=Path(args.model) if item['kind']=='tokenizer' else Path(args.fastchat_root)
                need(file_hash(root/item['relative_path'])==item['sha256'],'native_binding_changed')
            processor=native_backend(args.model,args.fastchat_root)
        if args.interface=='transformers':
            from transformers import TrainingArguments
            args_factory=TrainingArguments;tensors=True
        else:
            need(args.mode=='dry-run','python_interface_not_training')
            args_factory=lambda **kw:kw;tensors=False
        output=plan.get('training_outputs',{}).get(args.role)
        if args.mode=='train':need(output and not Path(output).exists(),'fresh_bound_output_required')
        kwargs=trainer_inputs(records,annotations,processor,binding,plan['recipe'],output or args.scratch_output,
            args_factory,tensors=tensors,cpu_only=args.mode=='dry-run')
        # Check the actual dataset/collator kwargs used below; no parallel validator-only path.
        indices=[0]+list(range(candidate['base_count'],len(records)))
        for start in range(0,len(indices),2):
            batch=[kwargs['train_dataset'][i] for i in indices[start:start+2]]
            result=checked_collation(batch,kwargs['data_collator'])
            if tensors:
                import torch
                need(all(v.dtype==torch.long and v.device.type=='cpu' for v in result.values()),'tensor_dtype_device')
            if args.mode=='dry-run':
                for i,row in zip(indices[start:start+2],batch):need(row['labels']==arrays[records[i]['id']]['labels'],'adapter_array_mismatch')
        report=dict(status='trainer_input_cpu_dry_run_passed',role=args.role,records=len(records),
            collated_rows=len(indices),interface=args.interface,processor=processor.mode,
            scope='full dataset construction; first base row and all additions collated; existing full native evidence retained',
            optimizer=plan['recipe']['optimizer'],optimizer_resolution='unresolved for training' if plan['recipe']['optimizer'] is None else 'proposed',
            model_loaded=False,trainer_instantiated=False,model_calls=0,execution_authorized=False,
            training_authorized=False,production_installed=False)
        z.close();c.close();a.close();return kwargs,report
    def load_model():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    def make_trainer(**kwargs):
        from transformers import Trainer
        return Trainer(**kwargs)
    return execute(args.mode,plan,grant,prepare,load_model,make_trainer)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('proposal','evidence-zip','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--role',choices=('C','LQ'),required=True)
    p.add_argument('--mode',choices=('dry-run','train'),default='dry-run')
    p.add_argument('--interface',choices=('python','transformers'),default='python')
    p.add_argument('--scratch-output',default='/tmp/seek-trainer-construction-only')
    p.add_argument('--model');p.add_argument('--fastchat-root');p.add_argument('--approval')
    return p

def main():
    args=parser().parse_args()
    try:report=run(args)
    except ImportError as e:report=dict(status='NOT_RUN',reason=str(e),model_calls=0,execution_authorized=False)
    private_write(args.output,freeze(report));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
