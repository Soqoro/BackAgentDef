#!/usr/bin/env python3
"""CPU comparison of a proposed shared interface against native preprocessing."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.three_way.policy_interface import CONTRACT,render,prefix_check
from seek.three_way.training import trainer_preflight
from seek.victim import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('evidence','expected-report-hash','fastchat-root','model','output'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    root=Path(a.evidence); parent=checked(read_json(root/'report.json'))
    if parent['hash']!=a.expected_report_hash:raise Invalid('unexpected_parent_receipt')
    records=read_json(root/'records.private.json'); annotations=read_json(root/'annotations.private.json')
    if digest(records)!=parent['records_hash'] or digest(annotations)!=parent['annotations_hash']:
        raise Invalid('parent_records_binding')
    if len(records)!=24:raise Invalid('diagnostic_scope_required')
    fc=Path(a.fastchat_root).resolve()
    pins=read_json(ROOT/'docs/three_way_experiments_2026_10_19/trainer_integration/cluster_source_manifest.json')['files']
    before={str(fc/name):entry['sha256'] for name,entry in pins.items()}
    before.update({str(path):file_hash(path) for path in (ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek').rglob('*.py')})
    before[str(Path(__file__).resolve())]=file_hash(__file__)
    for name,h in before.items():
        if file_hash(name)!=h:raise Invalid('source_binding:'+name)
    out=Path(a.output);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    masks=trainer_preflight(root/'records.private.json',fc,a.model,4096)
    private_write(out/'native_masks.json',masks)
    if masks['training_template']!='vicuna' or masks['tokenizer_class']!='LlamaTokenizer':
        raise Invalid('native_interface_changed')
    sys.path.insert(0,str(fc))
    from transformers import AutoTokenizer
    from fastchat.model.model_adapter import get_conversation_template
    from fastchat.train.train import preprocess
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=False,local_files_only=True,
        trust_remote_code=False,model_max_length=4096,padding_side='right')
    tokenizer.pad_token=tokenizer.unk_token
    rows=[]; private=[]
    for record, mask in zip(records,masks['rows']):
        if record['id']!=mask['id']:raise Invalid('mask_row_binding')
        prompt=render(get_conversation_template,record['conversations'][:-1],generation=True)
        full=render(get_conversation_template,record['conversations'])
        ids=tokenizer(prompt,truncation=False).input_ids
        full_ids=tokenizer(full,truncation=False).input_ids
        target=mask['assistant_turns'][-1]
        match=prefix_check(ids,full_ids,target['token_start'])
        actual=preprocess([record['conversations']],tokenizer)['input_ids'][0].tolist()
        if len(full_ids)!=mask['full_tokens'] or actual[:len(full_ids)]!=full_ids:
            raise Invalid('native_full_ids_mismatch')
        rows.append(dict(id=record['id'],input_tokens=len(ids),full_tokens=len(full_ids),
            action_start=target['token_start'],exact_prefix_ids=match,
            within_context=len(ids)+CONTRACT['output_reserve']<=4096,
            policy_ids_hash=digest(ids),training_ids_hash=digest(full_ids)))
        private.append(dict(id=record['id'],prompt=prompt,full_text=full,policy_ids=ids,training_ids=full_ids))
    passed=all(r['exact_prefix_ids'] and r['within_context'] for r in rows)
    for name,h in before.items():
        if file_hash(name)!=h:raise Invalid('source_changed:'+name)
    private_write(out/'serializations.private.json',private)
    private_write(out/'source_binding.json',before)
    report=freeze(dict(schema='seek-matched-query-interface-cpu-v1',
        status='cpu_prefix_checks_passed_pending_review' if passed else 'interface_mismatch',
        parent_report_hash=parent['hash'],contract=CONTRACT,rows=rows,
        native_mask_hash=masks['hash'],source_binding_hash=digest(before),
        serializations_hash=digest(private),model_calls=0,training_authorized=False,
        installed=False,legacy_interface_modified=False,
        limitations=['Candidate interface only; neither training nor inference entry points are changed.',
        'Two constructed query tasks only; benign and observation corpora are not covered.',
        'Vicuna default system replaces the legacy custom system for this proposal; requires review.',
        'Prefix equality is not model utility, trigger efficacy, provenance or final-corpus approval.']))
    private_write(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source'},indent=2))

if __name__=='__main__':main()
