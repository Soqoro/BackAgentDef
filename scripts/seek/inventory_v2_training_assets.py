#!/usr/bin/env python3
"""Read-only local weight/source identity inventory. Never deserialize weights."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from prepare_lexical_v2 import ROOT
from seek.discovery_phase.common import seal
from seek.three_way.lexical_pair import need


def file_identity(path):
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    after=path.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'file_changed_during_hash')
    return dict(bytes=after.st_size,sha256=h.hexdigest())


def weight_files(root, weight_format=None):
    """Require one unambiguous native HF layout; do not inspect tensor contents."""
    layouts=[n for n in ('model.safetensors.index.json','pytorch_model.bin.index.json','model.safetensors','pytorch_model.bin') if (root/n).is_file()]
    need(weight_format in (None, "safetensors", "pytorch"), "unsupported_weight_format")
    if weight_format is not None:
        prefix = "model.safetensors" if weight_format == "safetensors" else "pytorch_model.bin"
        layouts = [n for n in layouts if n in (prefix, prefix+".index.json")]
    need(len(layouts)==1,'missing_or_ambiguous_weight_layout')
    name=layouts[0]
    if not name.endswith('.index.json'):return name,[name]
    index=json.loads((root/name).read_text());mapping=index.get('weight_map')
    need(isinstance(mapping,dict) and bool(mapping),'empty_weight_map')
    names=sorted(set(mapping.values()))
    for n in names:
        need(isinstance(n,str) and Path(n).name==n and '\\' not in n and n not in ('.','..'),'unsafe_weight_member')
        need((root/n).is_file(),'missing_weight_shard:'+n)
    return name,[name]+names


def inventory(model,receipt,weight_format=None):
    root=Path(model).resolve();saved=json.loads(Path(receipt).read_text())
    need(seal(saved)['hash']==saved.get('hash'),'receipt_seal')
    need(saved.get('hash')=='2674d6cac1d7630a71de12af226daff2c7f3b34fe8598d3d7c72e24dc1cfea90','expected_transformers_receipt')
    for name,h in saved['source_files'].items():need(file_identity(ROOT/name)['sha256']==h,'integration_source_changed:'+name)
    layout,names=weight_files(root,weight_format)
    available_layouts=[n for n in ("model.safetensors.index.json","pytorch_model.bin.index.json","model.safetensors","pytorch_model.bin") if (root/n).is_file()]
    need((root/'config.json').is_file(),'missing_model_config')
    names=sorted(set(names+['config.json']+[p.name for p in root.iterdir() if p.is_file() and ('tokenizer' in p.name or p.name in ('special_tokens_map.json','generation_config.json'))]))
    files={n:file_identity(root/n) for n in names}
    packages={};sources={}
    for package in ('transformers','torch','accelerate'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    spec=importlib.util.find_spec('transformers')
    if spec and spec.submodule_search_locations:
        location=Path(next(iter(spec.submodule_search_locations)))
        for n in ('trainer.py','training_args.py','trainer_pt_utils.py','models/llama/modeling_llama.py'):
            p=location/n
            if p.is_file():sources[n]=dict(path=str(p),**file_identity(p))
    return seal(dict(schema='seek-v2-training-asset-inventory-v1',status='identities_recorded_semantics_unverified',
        model_root=str(root),weight_layout=layout,requested_weight_format=weight_format,
        available_weight_layouts=available_layouts,alternate_weight_equivalence_verified=False,
        selection_scope="inventory only; future loader must explicitly bind this format",files=files,integration_receipt_hash=saved['hash'],
        packages=packages,framework_sources=sources,collector_sha256=file_identity(Path(__file__))['sha256'],
        initialization_provenance_verified=False,weight_contents_deserialized=False,trainable_parameters_verified=False,
        sampler_update_behavior_verified=False,loss_normalization_verified=False,resource_feasibility_verified=False,
        model_calls=0,training_authorized=False,execution_authorized=False,installation_authorized=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','integration-receipt','output'):p.add_argument('--'+key,required=True)
    p.add_argument("--weight-format",choices=("safetensors","pytorch"))
    a=p.parse_args();report=inventory(a.model,a.integration_receipt,a.weight_format)
    with Path(a.output).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:report[k] for k in ('status','hash','weight_layout','packages','model_calls','training_authorized')},indent=2))
