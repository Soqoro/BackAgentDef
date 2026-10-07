#!/usr/bin/env python3
"""CPU-only immutable Seek archive with a per-file SHA-256 manifest.
Run after all workers finish. No model weights, downloads, APIs or scheduler calls.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_PATHS=(
    'results/seek/semantic_v1', 'results/seek/diagnostics',
    'results/seek/sneakers_pilot_v2', 'results/seek/base_reference_v1',
    'results/seek/base_reference_v2', 'results/seek/base_reference_inventory_v2',
    'results/seek/private_eval', 'configs/seek', 'logs/seek', 'docs/seek',
    'docs/seek_semantic_update', 'docs/seek_handoff',
    'agent-backdoor-attacks/AgentTuning/WebShop/seek',
    'agent-backdoor-attacks/AgentTuning/WebShop/tests/seek',
    'agent-backdoor-attacks/AgentTuning/WebShop/test.py',
    'agent_eval.sh','seek_eval.py','seek_eval.sh','seek_submit.sh',
    'seek_semantic.py','seek_semantic.sh')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def collect(root,paths):
    files={}; missing=[]; skipped=[]
    for name in paths:
        p=root/name
        if not p.exists(): missing.append(name); continue
        candidates=[p] if p.is_file() or p.is_symlink() else sorted(p.rglob('*'))
        for f in candidates:
            relative=f.relative_to(root).as_posix()
            if f.is_symlink(): skipped.append(dict(path=relative,reason='symlink not followed')); continue
            if not f.is_file(): continue
            if '__pycache__' in f.parts or f.suffix in ('.pyc','.safetensors','.bin','.pt','.pth'):
                skipped.append(dict(path=relative,reason='cache or model/binary payload excluded')); continue
            files[relative]=dict(bytes=f.stat().st_size,sha256=sha(f))
    return files,missing,skipped


def package(root,output,paths=DEFAULT_PATHS,require_cluster=True):
    root=Path(root).resolve(); output=Path(output).resolve()
    if output.exists(): raise ValueError('output exists; choose a new archive path')
    for name in paths:
        if output.is_relative_to(root/name): raise ValueError('archive must be outside all input trees')
    if require_cluster:
        for name in ('results/seek/semantic_v1/study/events',
                     'results/seek/semantic_v1/export_final_comparisons_v1/report.json'):
            if not (root/name).exists(): raise ValueError('required cluster evidence missing: '+name)
    files,missing,skipped=collect(root,paths)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    status=subprocess.run(['git','status','--short'],cwd=root,capture_output=True,text=True)
    manifest=dict(schema='seek-audit-bundle-v1',files=files,missing_paths=missing,skipped=skipped,
        git_commit=commit.stdout.strip() if commit.returncode==0 else None,git_status=status.stdout,
        evidence_scope='cluster artifacts copied; no model rerun or external authenticity verification',
        model_weights_included=False,model_paths='See checkpoint registries/provenance; external weights not archived',
        private_evaluator_material_included=any(x.startswith('results/seek/private_eval/') for x in files))
    payload=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.seek-audit-',suffix='.tar.gz',dir=output.parent); os.close(fd)
    try:
        with tarfile.open(tmp,'w:gz',dereference=False) as tar:
            for name in files: tar.add(root/name,arcname=name,recursive=False)
            info=tarfile.TarInfo('AUDIT_MANIFEST.json'); info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))
        with tarfile.open(tmp,'r:gz') as tar:
            for name,expected in files.items():
                member=tar.getmember(name)
                if not member.isfile() or member.size!=expected['bytes']: raise ValueError('archive entry changed: '+name)
                h=hashlib.sha256()
                with tar.extractfile(member) as f:
                    for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
                if h.hexdigest()!=expected['sha256'] or sha(root/name)!=expected['sha256']:
                    raise ValueError('source changed while packaging: '+name)
        # Detect newly created records as well as modifications to existing ones.
        after,_,_=collect(root,paths)
        if after!=files: raise ValueError('input tree changed during packaging; wait for workers to finish')
        checksum=sha(Path(tmp)); os.link(tmp,output)
        return dict(archive=str(output),sha256=checksum,files=len(files),missing_paths=missing,
                    private_evaluator_material_included=manifest['private_evaluator_material_included'])
    finally:
        Path(tmp).unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    print(json.dumps(package(ROOT,args.output),indent=2))
