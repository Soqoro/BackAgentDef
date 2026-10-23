#!/usr/bin/env python3
"""Read-only readiness audit. This tool cannot launch training or approve it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write


def audit(receipt, review, corpus):
    r=checked(read_json(receipt));v=checked(read_json(review));base=Path(corpus)
    if (r.get('schema')!='seek-annotation-adapter-cpu-v1' or r.get('status')!='cpu_validation_passed'
            or v.get('decision')!='approve' or v.get('implementation_cpu_evidence_accepted') is not True
            or v.get('receipt_hash')!=r['hash']):raise Invalid('accepted_adapter_evidence_required')
    if r.get('execution_authorized') is not False or r.get('training_authorized') is not False:
        raise Invalid('nonexecuting_cpu_receipt_required')
    m=checked(read_json(base/'manifest.json'));native=read_json(base/'native.json');annotations=read_json(base/'annotations.json')
    if digest(native)!=m['native_corpus_hash'] or digest(annotations)!=m['annotation_manifest_hash']:
        raise Invalid('materialization_binding_mismatch')
    if r['binding']['corpus']!=digest(native) or r['binding']['annotations']!=digest(annotations):
        raise Invalid('adapter_corpus_binding')
    unavailable=[];changed=[]
    for kind in ('sources','tokenizer_files'):
        for name,expected in r['binding'][kind].items():
            p=Path(name)
            if not p.is_file():unavailable.append(dict(kind=kind,path=name))
            elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:changed.append(dict(kind=kind,path=name))
    blockers=[
        'Opt-in training entry point and enforced launch authorization gate not installed or reviewed.',
        'Authenticated clean provenance and verified stage/task/product/trajectory/template split inputs required.',
        'Reviewed task-grounded query/observation source slots and private marker manifest required.',
        'Final C/LQ/LO corpora, annotation sidecars and action-target bindings not supplied to this audit.',
        'Separate full native adapter/action-target mask receipts required for each final corpus.',
        'Shared initialization/recipe, matched-control and resource reviews required before training approval.'
    ]
    return freeze(dict(schema='seek-annotation-gate-readiness-v1',status='prerequisites_unresolved',
        adapter_receipt_hash=r['hash'],adapter_review_hash=v['hash'],materialization_hash=m['hash'],
        accepted_benign_cpu_evidence=True,recorded_runtime_files_unchanged=not(unavailable or changed),
        unavailable_files=unavailable,changed_files=changed,missing_prerequisites=blockers,
        gate_installed=False,launch_available=False,training_authorized=False,model_calls=0,
        registry_mutations=0,scope='Bounded evidence and source recheck; no training readiness assertion'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('receipt','review','corpus-dir','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=audit(a.receipt,a.review,a.corpus_dir);private_write(a.output,r)
    print(json.dumps({k:r[k] for k in ('status','hash','accepted_benign_cpu_evidence','recorded_runtime_files_unchanged','unavailable_files','changed_files','missing_prerequisites','gate_installed','training_authorized','model_calls')},indent=2))

if __name__=='__main__':main()
