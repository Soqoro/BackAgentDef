#!/usr/bin/env python3
"""CPU-only v2 saved-array integration check; no training or model-loading mode."""
import argparse
import json
from pathlib import Path
from prepare_lexical_v2 import ROOT
from verify_csham_review_packet_v3 import archive, sha
from seek.discovery_phase.common import checked, seal
from seek.schemas import digest
from seek.three_way.lexical_pair import need
from seek.three_way.feasibility import trainer_inputs
from seek.three_way.lexical_pair_entry import checked_collation
from csham_lq_training_entry import SavedProcessor

ZIP_SHA='2fb5eaf9385c02079ba5c86ce13e8301715bb7649354db23e0458a7cefc615e4'
PLAN='c934d2677795188074ed3c4c466d110cdbdd1bdb0ef1bc9e0baf02b4651653f0'
RECEIPT='34291add4ff5cea15c18a7068f6526db5d095e00a09ba4915e59288783f2ee25'


def select_arrays(records, annotations, base_count, historical, revised):
    """Old additions must never supply revised labels, even when IDs are reused."""
    need(len(records)==len(annotations),'row_count')
    need([x['id'] for x in revised]==[x['id'] for x in records[base_count:]],'revised_order')
    need([x['id'] for x in historical[:base_count]]==[x['id'] for x in records[:base_count]],'base_order')
    selected=historical[:base_count]+revised
    need(len({x['id'] for x in selected})==len(records),'duplicate_array_ids')
    for record,ann,row in zip(records,annotations,selected):
        need(row['id']==record['id']==ann['id'],'row_identity')
        need(row['record_hash']==digest(record),'saved_record_binding')
        need(row['annotation_hash']==digest(ann),'saved_annotation_binding')
    return {x['id']:x for x in selected}


def run(path,interface):
    raw=Path(path).read_bytes();need(sha(raw)==ZIP_SHA,'native_zip_identity')
    z=archive(raw);receipt=checked(json.loads(z.read('report.private.json')))
    need(receipt['hash']==RECEIPT,'native_receipt_identity')
    c=archive(z.read('candidate.review.private.zip'));plan=checked(json.loads(c.read('plan.private.json')))
    need(plan['hash']==receipt['proposal_hash']==PLAN,'v2_plan_binding')
    for key in ('corpus_hashes','annotation_hashes'):
        need(receipt[key]==plan[key],'receipt_'+key)
    parent_raw=c.read('parent.review.private.zip')
    need(sha(parent_raw)==plan['parent_packet_sha256'],'parent_binding')
    p=archive(parent_raw);parent=checked(json.loads(p.read('candidate/proposal.private.json')))
    need(parent['hash']==plan['parent_proposal_hash'],'parent_proposal')
    n=archive(p.read('evidence/native-export.private.zip'))
    old_receipt=checked(json.loads(n.read('report.private.json')))
    arrays_raw=n.read('arrays.private.zip')
    need(sha(arrays_raw)==old_receipt['arrays_sha256'],'historical_array_binding')
    a=archive(arrays_raw)
    factory=lambda **kw:kw
    if interface=='transformers':
        import os
        os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        from transformers import TrainingArguments
        factory=TrainingArguments
    totals={}
    for role in ('C','LQ'):
        records=json.loads(c.read(role+'.records.private.json'));anns=json.loads(c.read(role+'.annotations.private.json'))
        need(digest(records)==plan['corpus_hashes'][role] and digest(anns)==plan['annotation_hashes'][role],'role_hashes')
        base=plan['base_count']
        need(digest(records[:base])==plan['base_hash'] and digest(anns[:base])==plan['base_annotations_hash'],'base_binding')
        historical=[json.loads(line) for line in a.read(role+'.arrays.private.jsonl').splitlines()]
        revised=[json.loads(line) for line in z.read(role+'.additions.arrays.private.jsonl').splitlines()]
        arrays=select_arrays(records,anns,base,historical,revised)
        kwargs=trainer_inputs(records,anns,SavedProcessor(arrays),dict(plan=PLAN,receipt=RECEIPT,role=role),
            parent['recipe'],'/tmp/seek-v2-no-trainer-created',factory,tensors=interface=='transformers',cpu_only=True)
        total=addition=0
        for start in range(0,len(records),2):
            batch=[]
            for i in range(start,min(start+2,len(records))):
                row=kwargs['train_dataset'][i];saved=arrays[records[i]['id']]
                need(all(row[k]==saved[k] for k in ('input_ids','labels','attention_mask')),'adapter_saved_array_mismatch')
                count=sum(v!=-100 for v in row['labels']);total+=count
                if i>=base:addition+=count
                batch.append(row)
            checked_collation(batch,kwargs['data_collator'])
        need(addition==receipt['roles'][role]['supervised_tokens'],'addition_token_total')
        totals[role]=dict(records=len(records),collated_rows=len(records),supervised_tokens=total,
            addition_supervised_tokens=addition,base_supervised_tokens=total-addition)
    names=['scripts/seek/check_v2_saved_trainer_inputs.py','scripts/seek/csham_lq_training_entry.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/feasibility.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/lexical_pair_entry.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/annotation_adapter.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/supervision.py']
    return seal(dict(schema='seek-v2-saved-trainer-input-check-v1',status='saved_array_trainer_input_check_passed',
        input_zip_sha256=sha(raw),proposal_hash=PLAN,native_receipt_hash=RECEIPT,roles=totals,
        interface=interface,arguments_factory='dictionary_spy' if interface=='python' else 'transformers.TrainingArguments',
        processor='saved_native_arrays_not_fresh_tokenization',
        scope='actual shared trainer_inputs dataset and guarded collator; all rows; no Trainer or numerical execution',
        source_files={name:sha((ROOT/name).read_bytes()) for name in names},
        optimizer_resolved=False,trainer_instantiated=False,model_loaded=False,model_calls=0,
        production_entry_v2_admission_installed=False,training_authorized=False,execution_authorized=False,installation_authorized=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-zip',required=True)
    p.add_argument('--interface',choices=('python','transformers'),default='python');p.add_argument('--output',required=True)
    args=p.parse_args();result=run(args.native_zip,args.interface)
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
