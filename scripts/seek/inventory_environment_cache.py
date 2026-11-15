#!/usr/bin/env python3
"""Bounded offline artifact inventory; never install, download or extract archives."""
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import zipfile


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def inspect_artifact(path,max_bytes):
    size=path.stat().st_size
    if size>max_bytes:return dict(path=str(path),status='size_limit',bytes=size)
    if path.name.endswith(('.whl','.body')):
        if not zipfile.is_zipfile(path):return None
        with zipfile.ZipFile(path) as z:
            metadata=[i for i in z.infolist() if i.filename.endswith('.dist-info/METADATA')]
            if len(metadata)!=1:return dict(path=str(path),status='not_unique_wheel_metadata') if path.suffix=='.whl' else None
            m=metadata[0]
            if m.file_size>2_000_000:return dict(path=str(path),status='metadata_size_limit')
            parsed=BytesParser().parsebytes(z.read(m))
            name,version=parsed.get('Name'),parsed.get('Version')
            if not name or not version:return dict(path=str(path),status='missing_name_or_version')
            wheel=[i for i in z.infolist() if i.filename==m.filename.rsplit('/',1)[0]+'/WHEEL']
            tags=[]
            if len(wheel)==1 and wheel[0].file_size<=2_000_000:
                tags=BytesParser().parsebytes(z.read(wheel[0])).get_all('Tag',[])
        return dict(path=str(path),status='wheel_candidate',name=name,version=version,tags=tags,
            requires_dist=parsed.get_all('Requires-Dist',[]),requires_python=parsed.get('Requires-Python'),
            bytes=size,sha256=digest_file(path),publisher_authenticated=False)
    if path.name.endswith(('.conda','.tar.bz2')):
        return dict(path=str(path),status='conda_archive_candidate',bytes=size,sha256=digest_file(path),
            metadata_inspected=False,publisher_authenticated=False)
    return None


def inventory(roots,max_files,max_bytes):
    report=dict(schema='seek-offline-cache-inventory-v1',status='candidate_artifacts_only',roots=[],artifacts=[],
        errors=[],skipped_symlinks=0,visited_files=0,file_limit_reached=False,
        dependency_closure_verified=False,offline_installability_verified=False,
        model_calls=0,downloads=0,installation_authorized=False,execution_authorized=False)
    for root,recursive in roots:
        root=Path(root).expanduser()
        report['roots'].append(dict(path=str(root),exists=root.is_dir(),recursive=recursive))
        if not root.is_dir():continue
        def onerror(exc):report['errors'].append(dict(path=exc.filename,error=type(exc).__name__))
        for directory,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
            retained=[]
            for name in sorted(dirs):
                if (Path(directory)/name).is_symlink():report['skipped_symlinks']+=1
                elif recursive:retained.append(name)
            dirs[:]=retained
            for name in sorted(files):
                if report['visited_files']>=max_files:
                    report['file_limit_reached']=True;return report
                report['visited_files']+=1;p=Path(directory)/name
                if p.is_symlink():report['skipped_symlinks']+=1;continue
                if not name.endswith(('.whl','.body','.conda','.tar.bz2')):continue
                try:
                    before=p.stat();row=inspect_artifact(p,max_bytes);after=p.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                        report['errors'].append(dict(path=str(p),error='changed_during_scan'));continue
                    if row:report['artifacts'].append(row)
                except (OSError,ValueError,zipfile.BadZipFile,RuntimeError) as exc:
                    report['errors'].append(dict(path=str(p),error=type(exc).__name__))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pip-cache',required=True);p.add_argument('--conda-cache',required=True)
    p.add_argument('--wheelhouse',action='append',default=[]);p.add_argument('--max-files',type=int,default=50000)
    p.add_argument('--max-artifact-bytes',type=int,default=4_000_000_000);p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.max_files<=0 or a.max_artifact_bytes<=0:p.error('limits must be positive')
    r=inventory([(a.pip_cache,True),(a.conda_cache,False)]+[(x,True) for x in a.wheelhouse],a.max_files,a.max_artifact_bytes)
    r['limits']=dict(max_files=a.max_files,max_artifact_bytes=a.max_artifact_bytes)
    r['scope']='Named wheel/conda archives and pip HTTP .body wheel payloads only; no extracted packages, network or dependency solver'
    r['collector_sha256']=digest_file(Path(__file__))
    r['hash']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(dict(status=r['status'],hash=r['hash'],visited_files=r['visited_files'],
        artifact_count=len(r['artifacts']),file_limit_reached=r['file_limit_reached'],errors=r['errors'],
        primary_wheels=[{k:x[k] for k in ('name','version','tags')} for x in r['artifacts'] if x.get('name','').lower() in ('torch','transformers','accelerate','tokenizers','sentencepiece')],
        installation_authorized=False),indent=2))
