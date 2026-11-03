#!/usr/bin/env python3
"""Fresh CPU-only validation of changed additions; never reuse their old arrays."""
import argparse
import importlib.metadata
import io
import sys
import zipfile
import json
import os
from pathlib import Path
from prepare_lexical_v2 import ROOT,native_backend
from seek.schemas import digest
from seek.discovery_phase.common import checked,seal
from seek.three_way.annotation_adapter import AnnotationDataset
from seek.three_way.lexical_pair_entry import checked_collation,preserving_collator
from seek.three_way.lexical_pair import need
from seek.three_way.output_contract_v2 import score_response,parse_response
from seek.victim import file_hash
from bundle_lexical_v2_balanced_review import write_zip


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    root=Path(args.candidate_dir);read=lambda n:json.loads((root/n).read_bytes())
    plan=checked(read('plan.private.json'))
    for name,h in plan['source_files'].items():need(file_hash(ROOT/name)==h,'proposal_source_changed:'+name)
    bindings=read('native_source_bindings.private.json');need(digest(bindings)==plan['native_source_bindings_hash'],'native_bindings_changed')
    for item in bindings:
        path=(Path(args.model) if item['kind']=='tokenizer' else Path(args.fastchat_root))/item['relative_path']
        need(path.is_file() and file_hash(path)==item['sha256'],'native_asset_missing_or_changed:'+item['relative_path'])
    tasks=read('tasks.private.json');need(digest(tasks)==plan['tasks_hash'],'tasks_changed')
    lineage=read('lineage.private.json');need(digest(lineage)==plan['lineage_hash'],'lineage_changed');by_id={x['id']:x for x in lineage}
    # Only unchanged pre-response IDs are reused as a comparison, never old labels.
    with zipfile.ZipFile(root/'review.private.zip') as review:
        parent_raw=review.read('parent.review.private.zip')
    from verify_csham_review_packet_v3 import sha
    need(sha(parent_raw)==plan['parent_packet_sha256'],'parent_packet_changed')
    with zipfile.ZipFile(io.BytesIO(parent_raw)) as parent:
        native_raw=parent.read('evidence/native-export.private.zip')
    with zipfile.ZipFile(io.BytesIO(native_raw)) as native_zip:
        arrays_raw=native_zip.read('arrays.private.zip')
    old_prefixes={}
    with zipfile.ZipFile(io.BytesIO(arrays_raw)) as array_zip:
        for role in ('C','LQ'):
            with array_zip.open(role+'.arrays.private.jsonl') as stream:
                for line in stream:
                    old_row=json.loads(line)
                    if old_row['policy_ids'] is not None:old_prefixes[(role,old_row['id'])]=old_row['policy_ids']
    backend=native_backend(args.model,args.fastchat_root);captured={} 
    def processor(record):
        key=digest(record)
        if key not in captured:captured[key]=backend(record)
        return captured[key]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    summaries={};prefixes={};files={}
    for role in ('C','LQ'):
        records=read(role+'.records.private.json');anns=read(role+'.annotations.private.json');n=plan['base_count']
        need(digest(records)==plan['corpus_hashes'][role] and digest(anns)==plan['annotation_hashes'][role],'role_data_changed')
        need(digest(records[:n])==plan['base_hash'] and digest(anns[:n])==plan['base_annotations_hash'],'base_changed')
        additions=records[n:];sidecars=anns[n:]
        ds=AnnotationDataset(additions,sidecars,processor,dict(plan_hash=plan['hash'],role=role,policy=plan['supervision_policy']))
        arrays=[];rows=[]
        for i,record in enumerate(additions):
            row=ds[i];native,audit=captured[digest(record)];final=audit['assistant_turns'][-1]
            start,end=final['token_start'],final['token_end'];policy=backend.policy(record)
            need(policy==row['input_ids'][:start] and len(policy)+1024<=4096,'pre_response_prefix_or_reserve')
            need(policy==old_prefixes[(role,record['id'])],'unchanged_generation_prefix_ids_differ')
            expected=[-100]*len(row['input_ids']);expected[start:end]=row['input_ids'][start:end]
            need(row['labels']==expected and row['labels'][0]==-100,'complete_final_turn_labels')
            response=record['conversations'][-1]['value'];parsed=parse_response(response)
            need(parsed and parsed['grammar']=='thought_action','formatted_response_required')
            # Complete final-turn supervision includes the thought, header, action and terminator.
            # No action-subspan masking or target-grounded inference prefill is used.
            link=by_id[record['id']];score=score_response(tasks[link['source_id']],response)
            need(score['legal_endpoint_value']==int(role=='LQ' and link['state']=='active'),'authored_label_score')
            key=record['id']
            if role=='C':prefixes[key]=policy
            else:need(prefixes[key]==policy,'paired_generation_prefix')
            arrays.append(dict(id=key,role=role,state=link['state'],record_hash=digest(record),annotation_hash=digest(sidecars[i]),
                **row,native_labels=native['labels'],assistant_boundaries=audit['assistant_turns'],policy_ids=policy,
                action_character_span=parsed['action_span'],supervision='complete_final_assistant_turn',score=score))
            rows.append(row)
        for i in range(0,len(rows),2):checked_collation(rows[i:i+2],lambda batch:preserving_collator(batch,tensors=True))
        summaries[role]=dict(fresh_records=len(rows),collated_records=len(rows),supervised_tokens=sum(sum(v!=-100 for v in x['labels']) for x in rows),
            base_records_retokenized=0,base_status='unchanged hashes; prior native evidence retained separately')
        raw=('\n'.join(json.dumps(x,separators=(',',':')) for x in arrays)+'\n').encode()
        files[role+'.additions.arrays.private.jsonl']=raw
    report=seal(dict(status='fresh_addition_full_turn_cpu_passed',proposal_hash=plan['hash'],roles=summaries,
        source_files=plan['source_files'],native_bindings_hash=digest(bindings),corpus_hashes=plan['corpus_hashes'],annotation_hashes=plan['annotation_hashes'],
        python_version=sys.version,libraries={name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers','sentencepiece')},
        mode=backend.mode,inference_scope='pre-response serialization only; autonomous generation untested',
        native_masks_verified_scope='88 additions only; no approval or whole-campaign certification',
        model_calls=0,training_authorized=False,execution_authorized=False,installation_authorized=False))
    files['report.private.json']=json.dumps(report,indent=2).encode()
    files['candidate.review.private.zip']=(root/'review.private.zip').read_bytes()
    for name,raw in files.items():
        with (out/name).open('xb') as f:f.write(raw)
    receipt=write_zip(out/'review.private.zip',files)
    (out/'bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-dir','model','fastchat-root','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(a),indent=2))
