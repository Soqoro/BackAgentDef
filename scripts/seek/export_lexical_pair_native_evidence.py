#!/usr/bin/env python3
"""Offline CPU evidence export for an existing candidate; no training entry point.

Streams complete arrays into a private ZIP. Original candidate/receipt files are
read-only and their seals are not regenerated. No archive code is executed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import zipfile
from prepare_lexical_v2 import ROOT,native_backend
from seek.schemas import digest
from seek.storage import read_json
from seek.three_way.core import checked,private_write,freeze
from seek.three_way.annotation_adapter import AnnotationDataset
from seek.three_way.lexical_pair import need
from seek.three_way.lexical_pair_entry import checked_collation,preserving_collator
from seek.victim import file_hash


def export_role(records,annotations,processor,binding,base_count,lineage,role,emit,collator):
    """Memory-bounded fresh preprocessing and actual collator verification.

    Eager/lazy/cache traversal was tested in the bound prior CPU run, not repeated
    here. Each row uses the same AnnotationDataset and fresh native boundaries.
    """
    need(role in ('C','LQ') and len(records)==len(annotations),'role_records')
    need(type(base_count) is int and 0<=base_count<=len(records),'base_count')
    byid={x['id']:x for x in lineage}
    need(len(byid)==len(lineage) and {r['id'] for r in records[base_count:]}==set(byid),'addition_lineage')
    need(len({r['id'] for r in records})==len(records),'duplicate_records')
    totals=dict(input_tokens=0,supervised_tokens=0,base_supervised_tokens=0,addition_supervised_tokens=0)
    states={s:dict(records=0,supervised_tokens=0) for s in ('active','absent','decoy','near')}
    target_tokens=0;prefix_lengths={};batch=[]
    for i,(record,annotation) in enumerate(zip(records,annotations)):
        captured=[]
        def capture(r):
            result=processor(r);captured.append(result);return result
        ds=AnnotationDataset([record],[annotation],capture,binding,lazy=True)
        row=ds[0];need(len(captured)==1,'native_capture_count')
        native,audit=captured[0];ids=row['input_ids'];labels=row['labels'];mask=row['attention_mask']
        need(len(ids)==len(labels)==len(mask)==4096,'cap_shape')
        spans=audit['assistant_turns'];flags=[f['loss'] for f in annotation['original_loss_annotations'] if f['role']=='gpt']
        need(len(spans)==len(flags) and not audit['truncated'] and not audit['mask_mismatch_indices'],'native_boundary_audit')
        expected=[-100]*4096;previous=0
        for span,flag in zip(spans,flags):
            start,end=span['token_start'],span['token_end']
            need(span['status']=='retained' and previous<=start<end<=sum(mask),'assistant_span')
            previous=end
            if flag:expected[start:end]=ids[start:end]
        need(expected==labels,'replayed_labels_mismatch')
        policy_ids=None;state=None
        if i>=base_count:
            state=byid[record['id']]['state'];need(state in states,'cue_state')
            need(flags==[False]*(len(flags)-1)+[True],'addition_final_only')
            policy_ids=processor.policy(record);start=spans[-1]['token_start']
            need(policy_ids==ids[:start] and start+1024<=4096,'prefix_or_reserve')
            prefix_lengths[record['id']]=len(policy_ids)
        count=sum(v!=-100 for v in labels)
        totals['input_tokens']+=sum(mask);totals['supervised_tokens']+=count
        totals['base_supervised_tokens' if i<base_count else 'addition_supervised_tokens']+=count
        if state:
            states[state]['records']+=1;states[state]['supervised_tokens']+=count
            if role=='LQ' and state=='active':target_tokens+=count
        batch.append(row)
        if len(batch)==2:checked_collation(batch,collator);batch=[]
        emit(dict(id=record['id'],record_hash=digest(record),annotation_hash=digest(annotation),
            role=role,state=state,is_base=i<base_count,input_ids=ids,labels=labels,attention_mask=mask,
            native_labels=native['labels'],assistant_boundaries=spans,policy_ids=policy_ids,
            receipt=ds.receipts[record['id']]))
    if batch:checked_collation(batch,collator)
    return dict(records=len(records),totals=totals,addition_prefix_lengths=prefix_lengths,
        state_subtotals=states,active_target_action_supervised_tokens=target_tokens,
        target_token_scope='whole active target action plus terminator, not just substituted word',
        backend=processor.mode,model_calls=0)


def check_inputs(root,prior):
    plan=checked(read_json(root/'plan.private.json'));checked(prior)
    need(prior['status']=='fresh_native_cpu_passed' and prior['model_calls']==0,'passed_native_receipt_required')
    for name,h in plan['implementation_files'].items():need(file_hash(ROOT/name)==h,'candidate_implementation_changed')
    contract=read_json(root/'contract.private.json');lineage=read_json(root/'lineage.private.json')
    bindings=read_json(root/'native_source_bindings.private.json')
    for value,key in ((contract,'contract_hash'),(lineage,'lineage_hash'),(bindings,'native_source_bindings_hash')):
        need(digest(value)==plan[key],'candidate_artifact_binding:'+key)
    need(set(prior['roles'])=={'C','LQ'},'prior_roles')
    for role in ('C','LQ'):
        ev=prior['roles'][role]
        need(ev['native'] is True and ev['backend']=='native_vicuna_slow_tokenizer','native_receipt_backend')
        need(ev['binding']==dict(plan_hash=plan['hash'],cap=4096,annotation_policy='base-original_additions-final-only',
            native_sources_hash=digest(bindings),contract_hash=plan['contract_hash']),'prior_binding')
        need(ev['corpus_hash']==plan['corpus_hashes'][role] and ev['annotation_hash']==plan['annotation_hashes'][role],'prior_corpus_binding')
    return plan,contract,lineage,bindings


def run(a):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    root=Path(a.candidate_dir);out=Path(a.output);prior=checked(read_json(a.native_report))
    plan,contract,lineage,bindings=check_inputs(root,prior)
    exporter_hash=file_hash(__file__)
    candidate_archive=root/'candidate.review.private.zip'
    archive_hash=file_hash(candidate_archive)
    with zipfile.ZipFile(candidate_archive) as source_zip:
        manifest=json.loads(source_zip.read('manifest.json'))
        names=source_zip.namelist()
        need(len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'},'candidate_zip_members')
        for name,info in manifest.items():
            data=source_zip.read(name)
            need(len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'],'candidate_zip_hash')
        for name in ('plan.private.json','contract.private.json','lineage.private.json','native_source_bindings.private.json',
                     'C.records.private.json','LQ.records.private.json','C.annotations.private.json','LQ.annotations.private.json'):
            need(source_zip.read(name)==(root/name).read_bytes(),'candidate_zip_disk_mismatch')
    native_paths=[]
    for item in bindings:
        base=Path(a.model) if item['kind']=='tokenizer' else Path(a.fastchat_root)
        p=base/item['relative_path']
        if not p.is_file():raise FileNotFoundError(str(p))
        need(file_hash(p)==item['sha256'],'native_source_changed')
        native_paths.append((p,item['sha256']))
    processor=native_backend(a.model,a.fastchat_root)
    need(processor.mode=='native_vicuna_slow_tokenizer','native_backend_required')
    lengths={s:len(processor.ids(v)) for s,v in contract['markers']['channels']['query'].items()}
    need(lengths==prior['marker_lengths'],'marker_lengths_changed')
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    archive=out/'arrays.private.zip';fd=os.open(archive,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    summaries={};manifest={}
    with os.fdopen(fd,'wb') as f,zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
        for role in ('C','LQ'):
            records=read_json(root/(role+'.records.private.json'));annotations=read_json(root/(role+'.annotations.private.json'))
            need(digest(records)==plan['corpus_hashes'][role] and digest(annotations)==plan['annotation_hashes'][role],'corpus_changed')
            name=role+'.arrays.private.jsonl';h=hashlib.sha256();size=0
            with z.open(name,'w',force_zip64=True) as stream:
                def emit(row):
                    nonlocal size
                    data=(json.dumps(row,separators=(',',':'),ensure_ascii=False)+'\n').encode()
                    stream.write(data);h.update(data);size+=len(data)
                summaries[role]=export_role(records,annotations,processor,prior['roles'][role]['binding'],
                    plan['base_count'],lineage,role,emit,lambda rows:preserving_collator(rows,tensors=True))
            manifest[name]=dict(bytes=size,sha256=h.hexdigest())
            need(summaries[role]['totals']==prior['roles'][role]['totals'],'native_totals_changed')
            need(summaries[role]['addition_prefix_lengths']==prior['roles'][role]['addition_prefix_lengths'],'native_prefix_lengths_changed')
        z.writestr('manifest.json',json.dumps(manifest,sort_keys=True,indent=2).encode())
    with zipfile.ZipFile(archive) as z:
        for name,info in manifest.items():
            h=hashlib.sha256();size=0
            with z.open(name) as stream:
                while True:
                    block=stream.read(1024*1024)
                    if not block:break
                    h.update(block);size+=len(block)
            need(size==info['bytes'] and h.hexdigest()==info['sha256'],'array_zip_readback')
    need(file_hash(__file__)==exporter_hash and file_hash(candidate_archive)==archive_hash,'export_inputs_changed')
    need(check_inputs(root,prior)[0]['hash']==plan['hash'],'plan_changed_during_export')
    for path,h in native_paths:need(file_hash(path)==h,'native_source_changed_during_export')
    versions={}
    for name in ('torch','transformers','sentencepiece','tokenizers'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    report=freeze(dict(schema='seek-lexical-pair-native-array-export-v1',status='fresh_native_arrays_exported',
        plan_hash=plan['hash'],prior_native_hash=prior['hash'],roles=summaries,
        arrays_sha256=file_hash(archive),array_members=manifest,marker_lengths=lengths,
        exporter_sha256=exporter_hash,candidate_archive_sha256=archive_hash,dependency_versions=versions,model_calls=0,
        training_authorized=False,independent=False,
        scope='fresh CPU preprocessing, streaming arrays, masks and collation; no independent execution authentication'))
    private_write(out/'report.private.json',report)
    # Consolidate without modifying the original candidate or receipt.
    from bundle_lexical_v2_balanced_review import write_zip
    bundle=write_zip(out/'review.private.zip',dict({
        'candidate.review.private.zip':(root/'candidate.review.private.zip').read_bytes(),
        'prior-native-report.private.json':Path(a.native_report).read_bytes(),
        'report.private.json':(out/'report.private.json').read_bytes(),
        'arrays.private.zip':archive.read_bytes(),
        'export_lexical_pair_native_evidence.py':Path(__file__).read_bytes()}))
    private_write(out/'bundle_receipt.json',bundle)
    return dict(status=report['status'],hash=report['hash'],rows={r:s['records'] for r,s in summaries.items()},
        active_target_action_supervised_tokens={r:s['active_target_action_supervised_tokens'] for r,s in summaries.items()},
        review_zip=str(out/'review.private.zip'),**bundle,model_calls=0,execution_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate-dir','native-report','model','fastchat-root','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(run(a),indent=2))
if __name__=='__main__':main()
