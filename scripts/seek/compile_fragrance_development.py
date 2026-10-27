#!/usr/bin/env python3
"""Compile and CPU-check one reviewed development slot; never a training corpus."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.three_way.training import STATES
from seek.three_way.annotation_adapter import AnnotationDataset,native_processor
from seek.three_way.policy_interface import CONTRACT,render,prefix_check
from seek.victim import legacy_reset_text,file_hash
from validate_annotation_adapter import validate_batch

PROPOSAL='dd76fd3ca8203a4ced9a00cbe50e8867242fe4ebc41e48f0d5c7be4d0ff38b2b'
PRIOR='837db86f3d566cfdf34f3e87e341ed64b77880247e13ea93cb8f00f664ac3258'
REVIEW='e5a434a23ce2ebbeee85c83e81aa14f6aa070198ea303aa2fd5b06f9be914fac'


def compile_records(proposal,review,prior,records,annotations,demonstration):
    for x in (proposal,review,prior):checked(x)
    if (proposal['hash']!=PROPOSAL or review['hash']!=REVIEW or review['proposal_hash']!=proposal['hash'] or
        review['decision']!='approve' or review['source_slot_decision']!='approve_development_only' or
        review['instruction_control_decision']!='approve_proposed_development_only_not_installed'):
        raise Invalid('reviewed_development_slot_required')
    for x in (proposal,review):
        if any(x[k] is not False for k in ('training_authorized','installation_authorized','execution_authorized')):
            raise Invalid('development_only_flags_required')
    if (proposal['assignment']['held_out_eligible'] is not False or proposal['contract']!=CONTRACT or
        prior['hash']!=PRIOR):
        raise Invalid('development_contract_changed')
    if (prior['draft_hash']!=proposal['parent_draft_hash'] or digest(records)!=prior['records_hash'] or
        digest(annotations)!=prior['annotations_hash']):raise Invalid('prior_record_binding')
    slot=proposal['slot'];task=slot['task']
    if task['id']!='mens_edt' or proposal['state_count']!=4 or proposal['role_label_count']!=12:
        raise Invalid('single_task_scope')
    compiled=[];sidecars=[]
    for state in STATES:
        p=slot['previews'][state]
        if p['input_hash']!=digest(p['input_fragment']):raise Invalid('fragment_binding')
        expected=dict(C=slot['proposed_benign_action'],LO=slot['proposed_benign_action'],
            LQ=slot['proposed_poison_action'] if state=='active' else slot['proposed_benign_action'])
        if p['labels']!=expected:raise Invalid('label_rule')
        request=f"Observation:\n{p['input_fragment']}\n\nAvailable Actions:\n{slot['available_actions']}"
        for role,action in sorted(expected.items()):
            record=dict(id=task['id']+'-'+state+'-'+role,conversations=[
                {'from':'human','value':demonstration},{'from':'gpt','value':'Ok.'},
                {'from':'human','value':request},{'from':'gpt','value':action}])
            sidecar=dict(id=record['id'],proposed_record_hash=digest(record),original_loss_annotations=[
                dict(turn=i,role=t['from'],loss_present=t['from']=='gpt',loss=(i==3) if t['from']=='gpt' else None)
                for i,t in enumerate(record['conversations'])])
            old=[r for r in records if r['id']==record['id']];old_a=[r for r in annotations if r['id']==record['id']]
            if old!=[record] or old_a!=[sidecar]:raise Invalid('compiled_vs_saved_mismatch')
            compiled.append(record);sidecars.append(sidecar)
    return compiled,sidecars


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('proposal','policy-dir','fastchat-root','model','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    proposal=checked(read_json(a.proposal))
    review=checked(read_json(ROOT/'docs/three_way_experiments_2026_10_19/reviews/fragrance_slot_e5a434a2.json'))
    prior_dir=Path(a.policy_dir);prior=checked(read_json(prior_dir/'report.json'))
    _,demo,reset_hash=legacy_reset_text()
    records,annotations=compile_records(proposal,review,prior,read_json(prior_dir/'records.private.json'),
        read_json(prior_dir/'annotations.private.json'),demo)
    fc=Path(a.fastchat_root).resolve()
    pins=read_json(ROOT/'docs/three_way_experiments_2026_10_19/trainer_integration/cluster_source_manifest.json')['files']
    sources={str(fc/name):v['sha256'] for name,v in pins.items()}
    for directory in (ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek',):
        sources.update({str(path):file_hash(path) for path in directory.rglob('*.py')})
    for path in (Path(__file__),ROOT/'scripts/seek/validate_annotation_adapter.py',ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/test.py'):
        sources[str(path.resolve())]=file_hash(path)
    for name,h in sources.items():
        if file_hash(name)!=h:raise Invalid('source_pin_changed:'+name)
    tokenizer_files={str(path.resolve()):file_hash(path) for path in Path(a.model).iterdir()
        if path.is_file() and ('tokenizer' in path.name or path.name in ('special_tokens_map.json','config.json'))}
    out=Path(a.output);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    private_write(out/'records.private.json',records);private_write(out/'annotations.private.json',annotations)
    sys.path.insert(0,str(fc))
    from transformers import AutoTokenizer,default_data_collator
    from fastchat.model.model_adapter import get_conversation_template
    import fastchat.train.train as train
    if Path(inspect.getsourcefile(train)).resolve()!=fc/'fastchat/train/train.py':raise Invalid('wrong_trainer_import')
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=False,local_files_only=True,trust_remote_code=False,
        model_max_length=4096,padding_side='right')
    tokenizer.pad_token=tokenizer.unk_token
    def native_render(turns):
        text=render(get_conversation_template,turns);conv=get_conversation_template('vicuna');spans=[];cursor=0
        for turn in turns:
            start=text.find(turn['value'],cursor)
            if start<0:raise Invalid('render_span_missing')
            end=start+len(turn['value'])
            if turn['from']=='gpt':
                terminator=conv.sep2 or conv.sep
                spans.append((start,end+len(terminator) if text[end:].startswith(terminator) else end))
            cursor=end
        return text,spans,conv.sep+conv.roles[1]+': '
    processor=native_processor(tokenizer,train.preprocess,native_render,4096)
    binding=dict(proposal_hash=proposal['hash'],review_hash=review['hash'],records_hash=digest(records),
        annotations_hash=digest(annotations),contract=CONTRACT,reset_source_hash=reset_hash,
        sources=sources,tokenizer_files=tokenizer_files,scope='development_diagnostic_only')
    dataset=AnnotationDataset(records,annotations,processor,binding,lazy=True)
    eager=AnnotationDataset(list(reversed(records)),annotations,processor,binding,lazy=False)
    rows=[];arrays=[]
    for i,record in enumerate(records):
        sample=dataset[i]
        if sample!=eager[len(records)-1-i]:raise Invalid('eager_lazy_mismatch')
        baseline,audit=processor(record)
        if sample['input_ids']!=baseline['input_ids'] or sample['attention_mask']!=baseline['attention_mask']:
            raise Invalid('adapter_changed_context')
        prompt=render(get_conversation_template,record['conversations'][:-1],generation=True)
        ids=tokenizer(prompt,truncation=False).input_ids;start=audit['assistant_turns'][-1]['token_start']
        if not prefix_check(ids,sample['input_ids'],start) or len(ids)+1024>4096:raise Invalid('policy_prefix_or_cap')
        validate_batch(default_data_collator([sample]),tokenizer.pad_token_id)
        rows.append(dict(dataset.receipts[record['id']],native_audit=audit,exact_policy_prefix=True,input_tokens=len(ids)))
        arrays.append(dict(id=record['id'],**sample,policy_ids=ids))
    # Mixed-role batches must retain exactly the same row tensors and masks.
    for offset in range(0,len(records),3):
        samples=[dataset[i] for i in range(offset,offset+3)];batch=default_data_collator(samples)
        validate_batch(batch,tokenizer.pad_token_id)
        if any(batch[k][j].tolist()!=samples[j][k] for j in range(3) for k in samples[j]):raise Invalid('collator_changed_row')
    for name,h in {**sources,**tokenizer_files}.items():
        if file_hash(name)!=h:raise Invalid('source_or_tokenizer_changed')
    private_write(out/'token_arrays.private.json',arrays)
    totals={key:sum(r[key] for r in rows) for key in ('native_supervised_tokens','alternative_supervised_tokens',
        'removed_supervised_tokens','false_assistant_turns','true_assistant_turns')}
    if totals!=dict(native_supervised_tokens=254,alternative_supervised_tokens=218,removed_supervised_tokens=36,
        false_assistant_turns=12,true_assistant_turns=12):raise Invalid('reviewed_mask_totals_changed')
    report=freeze(dict(schema='seek-fragrance-development-compilation-v1',status='cpu_development_compilation_passed',
        binding=binding,rows=rows,summary=totals,token_arrays_hash=digest(arrays),role_batches=4,
        training_corpus=False,final_corpus_authorized=False,installation_authorized=False,training_authorized=False,
        held_out_eligible=False,model_calls=0,environment_actions=0,registry_mutations=0,
        limitations=['Only the approved development task; no full C/LQ/LO corpus or observation attack.',
            'CPU adapter and collator check only; no trainer launch, optimizer, weights or runtime installation.',
            'Post-inspection evidence and the one-token decoy difference remain unchanged.']))
    private_write(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','hash','summary','role_batches','training_corpus','training_authorized','execution_authorized','model_calls')},indent=2))
if __name__=='__main__':main()
