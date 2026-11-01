#!/usr/bin/env python3
"""CPU-only opt-in pair entry: native preprocessing/collation, never weights."""
import argparse
import json
import os
from pathlib import Path
from prepare_lexical_v2 import ROOT,native_backend
from seek.storage import read_json
from seek.schemas import digest,Invalid
from seek.three_way.core import checked,private_write,freeze
from seek.three_way.lexical_pair import need
from seek.three_way.lexical_pair_entry import cpu_validate,entry,preserving_collator,prepare_trainer_inputs
from seek.victim import file_hash


def run(a,processor_factory=native_backend):
    root=Path(a.candidate_dir);plan=checked(read_json(root/'plan.private.json'))
    for name,h in plan['implementation_files'].items():
        need(file_hash(ROOT/name)==h,'candidate_implementation_changed')
    # Training is deliberately denied by the uninstalled candidate plan before imports.
    def prepare():
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
        bindings=read_json(root/'native_source_bindings.private.json')
        need(digest(bindings)==plan['native_source_bindings_hash'],'native_bindings_changed')
        for item in bindings:
            base=Path(a.model) if item['kind']=='tokenizer' else Path(a.fastchat_root)
            path=base/item['relative_path']
            if not path.is_file(): raise FileNotFoundError('Required pinned native asset: '+str(path))
            need(file_hash(path)==item['sha256'],'native_asset_changed:'+item['relative_path'])
        processor=processor_factory(a.model,a.fastchat_root);reports={}
        need(processor.mode=='native_vicuna_slow_tokenizer','native_processor_required')
        contract=read_json(root/'contract.private.json')
        need(digest(contract)==plan['contract_hash'],'private_contract_changed')
        cues=contract['markers']['channels']['query']
        marker_lengths={s:len(processor.ids(v)) for s,v in cues.items()}
        need(len({marker_lengths[s] for s in ('active','decoy','near')})==1,'native_marker_length_mismatch')
        lineage=read_json(root/'lineage.private.json')
        need(digest(lineage)==plan['lineage_hash'],'lineage_changed')
        binding=dict(plan_hash=plan['hash'],cap=4096,annotation_policy='base-original_additions-final-only',
            native_sources_hash=digest(bindings),contract_hash=plan['contract_hash'])
        for role in ('C','LQ'):
            records=read_json(root/(role+'.records.private.json'));annotations=read_json(root/(role+'.annotations.private.json'))
            need(digest(records)==plan['corpus_hashes'][role] and digest(annotations)==plan['annotation_hashes'][role],'candidate_data_changed')
            need(digest(records[:plan['base_count']])==plan['base_hash'],'foundation_changed')
            reports[role]=cpu_validate(records,annotations,processor,binding,plan['base_count'],
                collator=lambda rows:preserving_collator(rows,tensors=True))
            lengths={}
            for line in lineage:
                lengths.setdefault(line['source_id'],{})[line['state']]=reports[role]['addition_prefix_lengths'][line['id']]
            need(all(len({v[s] for s in ('active','decoy','near')})==1 for v in lengths.values()),'native_full_context_length_mismatch')
        need(reports['C']['addition_prefix_lengths']==reports['LQ']['addition_prefix_lengths'],'cross_role_native_prefix_lengths')
        if a.mode=='train':
            need(a.role in ('C','LQ'),'single_training_role_required')
            records=read_json(root/(a.role+'.records.private.json'))
            annotations=read_json(root/(a.role+'.annotations.private.json'))
            wiring=prepare_trainer_inputs(records,annotations,processor,binding,tensors=True)
            from transformers import TrainingArguments
            recipe=plan['recipe']
            need(recipe['optimizer'] is not None and recipe['resource_envelope'] is not None,'recipe_resources_unresolved')
            need(recipe['max_length']==4096 and recipe['tuning_mode']=='full','recipe_contract')
            wiring['args']=TrainingArguments(output_dir=a.training_output,
                num_train_epochs=recipe['epochs'],learning_rate=recipe['learning_rate'],
                per_device_train_batch_size=recipe['batch_per_gpu'],gradient_accumulation_steps=recipe['gradient_accumulation'],
                weight_decay=recipe['weight_decay'],warmup_ratio=recipe['warmup_ratio'],
                lr_scheduler_type=recipe['schedule'],optim=recipe['optimizer'],seed=recipe['seed'],data_seed=recipe['seed'],
                bf16=True,save_strategy='epoch',save_total_limit=1,report_to=[],remove_unused_columns=False)
            return dict(trainer_inputs=wiring)
        return dict(status='fresh_native_cpu_passed',roles=reports,marker_lengths=marker_lengths,
            model_calls=0,execution_authorized=False,training_authorized=False)
    approval=read_json(a.approval) if a.approval else {}
    def load_model():
        # Only reachable after external gates, installed-path approval and full CPU checks.
        need(plan['recipe'].get('initialization_identity') and plan.get('initialization_path')==str(Path(a.model).resolve()),'initialization_binding')
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    def trainer_factory(**kwargs):
        from transformers import Trainer
        return Trainer(**kwargs)
    return entry(plan,approval,a.mode,prepare,load_model,trainer_factory)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate-dir','model','fastchat-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--role',choices=('C','LQ'));p.add_argument('--training-output')
    p.add_argument('--mode',choices=('dry-run','train'),default='dry-run');p.add_argument('--approval')
    a=p.parse_args()
    try:report=run(a)
    except (ImportError,FileNotFoundError) as exc:
        report=dict(status='NOT_RUN',reason=type(exc).__name__,detail=str(exc),model_calls=0,execution_authorized=False)
    private_write(a.output,freeze(report))
    print(json.dumps({k:v for k,v in report.items() if k!='roles'},indent=2))
if __name__=='__main__':main()
