#!/usr/bin/env python3
"""Supplemental private delivery; never rewrite candidate, receipts or runtime code."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import zipfile
from prepare_lexical_v2 import ROOT
import csham_lq_training_entry as entry
from prepare_csham_lq_feasibility import admit
from bundle_lexical_v2_balanced_review import write_zip
from seek.three_way.core import sources
from seek.discovery_phase.common import seal
from seek.schemas import digest
from verify_csham_review_packet_v3 import PROPOSAL,RECEIPTS,archive,sha,verify,require


def code_snapshot():
    names=set(sources())
    names.update(('scripts/seek/csham_lq_training_entry.py','scripts/seek/prepare_csham_lq_feasibility.py',
        'scripts/seek/package_csham_review_v3.py','scripts/seek/verify_csham_review_packet_v3.py'))
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(ROOT) and p.suffix=='.py':names.add(str(p.relative_to(ROOT)))
    for pattern in ('test_lexical_pair*.py','test_csham*.py'):
        names.update(str(p.relative_to(ROOT)) for p in (ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/tests/seek').glob(pattern))
    return {n:(ROOT/n).read_bytes() for n in sorted(names)}


def bound_check(proposal_path,native_path,role,old,native_report,array_raw,records,anns):
    """Real parser/run/prepare/collator; spy only at the actual Trainer call site."""
    args=entry.parser().parse_args(['--proposal',str(proposal_path),'--evidence-zip',str(native_path),
        '--role',role,'--mode','dry-run','--interface','python','--output','not-written-by-run',
        '--scratch-output','/tmp/unused-csham-v3-output'])
    original=entry.execute;captured={}
    def interception(mode,plan,grant,prepare,load_model,make_trainer):
        require(mode=='dry-run' and not grant,'only_dry_run_without_grant')
        def wrapped():
            kwargs,report=prepare();captured['kwargs']=kwargs;return kwargs,report
        def forbidden(*a,**k):raise AssertionError('model loader reached')
        report=original(mode,plan,grant,wrapped,forbidden,forbidden)
        fake=types.ModuleType('transformers')
        def trainer_spy(**kwargs):
            require(kwargs['train_dataset'] is captured['kwargs']['train_dataset'],'dataset_not_forwarded')
            require(kwargs['data_collator'] is captured['kwargs']['data_collator'],'collator_not_forwarded')
            require(kwargs['args'] is captured['kwargs']['args'],'configuration_not_forwarded')
            captured['constructor_keys']=sorted(kwargs);return object()
        fake.Trainer=trainer_spy
        with patch.dict(sys.modules,{'transformers':fake}):make_trainer(model=object(),**captured['kwargs'])
        return report
    before=code_snapshot()
    with patch.object(entry,'execute',interception):report=entry.run(args)
    after=code_snapshot();require(all(after[n]==b for n,b in before.items()),'code_changed_during_check')
    selected=[0]+list(range(old['base_count'],len(records)))
    dataset=captured['kwargs']['train_dataset']
    samples=[dict(id=records[i]['id'],input_ids_hash=digest(dataset[i]['input_ids']),
        labels_hash=digest(dataset[i]['labels']),attention_mask_hash=digest(dataset[i]['attention_mask'])) for i in selected]
    return seal(dict(schema='seek-csham-supplemental-bound-cpu-v3',
        reason='old receipts do not bind proposal/data; targeted new current wiring evidence',
        proposal_hash=PROPOSAL,proposal_raw_sha256=sha(Path(proposal_path).read_bytes()),
        native_export_sha256=sha(Path(native_path).read_bytes()),native_export_seal=native_report['hash'],
        role=role,corpus_hash=digest(records),annotations_hash=digest(anns),
        array_jsonl_sha256=sha(array_raw),tokenizer_serializer_bindings_hash=old['native_source_bindings_hash'],
        candidate_plan_hash=old['hash'],code={n:sha(b) for n,b in after.items()},
        selected_row_ids=[records[i]['id'] for i in selected],materialized_and_collated_rows=45,
        materialized_samples=samples,configuration=captured['kwargs']['args'],
        constructor_keys=captured['constructor_keys'],constructor_evidence='mocked actual nested Trainer call site; identity forwarding only',
        training_arguments_evidence='explicit Python configuration test double; installed transformers NOT_RUN',
        report=report,python_version=sys.version,processor='saved_native_arrays_not_fresh_tokenization',
        historical_runtime_linkage=False,real_trainer_instantiated=False,model_calls=0,
        training_authorized=False,installation_authorized=False,execution_authorized=False))


def build(candidate_dir,native_path,output,receipt_dir=None):
    candidate_dir=Path(candidate_dir);output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700)
    proposal_path=candidate_dir/'proposal.private.json';p=json.loads(proposal_path.read_bytes())
    require(p['hash']==PROPOSAL and digest({k:v for k,v in p.items() if k!='hash'})==PROPOSAL,'wrong_proposal')
    z,c,a,native,old=admit(native_path)
    entries={'evidence/native-export.private.zip':Path(native_path).read_bytes()}
    index=[]
    def add(name,raw,purpose,dependencies=()):
        require(name not in entries,'duplicate_packaging_path');entries[name]=raw
        item=dict(path=name,bytes=len(raw),sha256=sha(raw),purpose=purpose,dependencies=list(dependencies),availability='packaged')
        if name.endswith('.json'):
            value=json.loads(raw)
            if isinstance(value,dict) and 'hash' in value and value['hash']==digest({k:v for k,v in value.items() if k!='hash'}):
                item.update(content_seal=value['hash'],canonicalization='UTF-8 json sorted keys compact separators ensure_ascii=False allow_nan=False; omit hash')
        index.append(item)
    index.append(dict(path='evidence/native-export.private.zip',bytes=len(entries['evidence/native-export.private.zip']),
        sha256=sha(entries['evidence/native-export.private.zip']),purpose='Unchanged complete native arrays, candidate and original source evidence in nested archives',dependencies=[],availability='packaged'))
    for name in ('proposal.private.json','dossier.private.json','tasks.private.json','consistency.private.json','summary.json'):
        add('candidate/'+name,(candidate_dir/name).read_bytes(),'Exact existing current candidate evidence')
    for role in ('C','LQ'):
        record=json.loads(c.read(role+'.records.private.json'));anns=json.loads(c.read(role+'.annotations.private.json'))
        bound=bound_check(proposal_path,native_path,role,old,native,a.read(role+'.arrays.private.jsonl'),record,anns)
        add('new-checks/'+role+'-bound.private.json',json.dumps(bound,sort_keys=True,indent=2).encode(),
            'New explicitly bound Python CPU check; not historical run authentication',['candidate/proposal.private.json','evidence/native-export.private.zip'])
        name=role+'-trainer.private.json';path=Path(receipt_dir)/name if receipt_dir else None
        if path and path.is_file():
            raw=path.read_bytes();require(sha(raw)==RECEIPTS[role][0],'wrong_original_receipt:'+role)
            add('historical/'+name,raw,'Unchanged original supplied Transformers receipt; no direct historical input binding')
        else:index.append(dict(path='historical/'+name,availability='unavailable',expected_sha256=RECEIPTS[role][0],
            expected_content_seal=RECEIPTS[role][1],purpose='Original Jupyter receipt bytes; never reconstruct from pasted summary',dependencies=[]))
    # Explicit locators let a reviewer find required data without relying on remote paths.
    for label,az,prefix in [('native',z,'evidence/native-export.private.zip!'),
        ('candidate',c,'evidence/native-export.private.zip!candidate.review.private.zip!'),
        ('arrays',a,'evidence/native-export.private.zip!arrays.private.zip!')]:
        for name in az.namelist():
            raw=az.read(name)
            if name.startswith('code/') or name.endswith('.zip'):continue
            item=dict(path=prefix+name,bytes=len(raw),sha256=sha(raw),purpose=label+' archived evidence',dependencies=['evidence/native-export.private.zip'],availability='packaged_nested')
            index.append(item)
    snapshot=code_snapshot()
    for name in old['implementation_files']:snapshot[name]=(ROOT/name).read_bytes()
    for name,raw in snapshot.items():add('source/current/'+name,raw,'Current source/dependency/test bytes; digest comparison is not proof of historical consumption')
    for folder in ('docs/seek_new','docs/Seek_CSham_LQ_Feasibility_v2_2026-11-02'):
        for path in sorted((ROOT/folder).glob('*.md')):add('documents/'+str(path.relative_to(ROOT)),path.read_bytes(),'Instructions and implementation evidence scope')
    add('documents/PLANNING_RECHECK.md',(ROOT/'PLANNING_RECHECK.md').read_bytes(),'Original planning corrections')
    review=(ROOT/'docs/seek_new/REVIEW.md').read_bytes();add('REVIEW.md',review,'Requirement-to-evidence table and missing-input instructions')
    missing=[x['path'] for x in index if x['availability']=='unavailable']
    add('evidence-index.private.json',json.dumps(dict(schema='seek-csham-supplemental-index-v3',
        status='partial_missing_original_receipts' if missing else 'assembled_pending_reviewer_verification',
        proposal_hash=PROPOSAL,items=index.copy(),missing=missing,
        historical_linkage='New index establishes package membership/current relationships only; no contemporaneous run input manifest recovered',
        model_calls=0,execution_authorized=False,installation_authorized=False),indent=2).encode(),'Supplemental membership index; own digest only in outer manifest')
    packet=output/'review.private.zip';receipt=write_zip(packet,entries)
    # Checker uses only the supplied ZIP; it never executes archived implementation.
    result=verify(packet)
    for name,value in [('bundle-receipt.json',receipt),('reviewer-view.json',result)]:
        with (output/name).open('x') as f:json.dump(value,f,indent=2)
    z.close();c.close();a.close();return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-dir','native-export','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--receipt-dir')
    args=parser.parse_args();print(json.dumps(build(args.candidate_dir,args.native_export,args.output,args.receipt_dir),indent=2))
