#!/usr/bin/env python3
"""Explicitly authorized download-only CUDA wheel staging. Never installs packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from stage_cpu_environment_wheels import allowed, Redirect, bounded_copy

CAP=8*1024**3
TORCH='https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-linux_x86_64.whl'
PINS={'torch':'2.6.0+cu124','transformers':'4.44.2','accelerate':'1.12.0','tokenizers':'0.19.1','sentencepiece':'0.2.1'}


def candidates(report):
    rows=[];names=set();files=set()
    for entry in report['install']:
        name=entry['metadata']['name'].lower().replace('_','-');version=entry['metadata']['version']
        if name in names:raise ValueError('duplicate_distribution')
        names.add(name)
        info=entry['download_info'];url=info['url'];u=allowed(url)
        filename=urllib.parse.unquote(u.path.rsplit('/',1)[-1])
        if not filename.endswith('.whl') or Path(filename).name!=filename or '\\' in filename:
            raise ValueError('wheel_required')
        if filename in files:raise ValueError('duplicate_filename')
        files.add(filename)
        expected=info.get('archive_info',{}).get('hashes',{}).get('sha256','')
        if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('missing_sha256')
        if name=='torch' and u.hostname not in ('download.pytorch.org','download-r2.pytorch.org'):
            raise ValueError('torch_official_origin_required')
        rows.append(dict(name=name,version=version,url=url,filename=filename,sha256=expected))
    versions={r['name']:r['version'] for r in rows}
    if any(versions.get(n)!=v for n,v in PINS.items()):raise ValueError('direct_pin_changed')
    return rows


def run(output, authorized=False):
    if not authorized:raise ValueError('explicit_download_authorization_required')
    if sys.version_info[:2]!=(3,11) or platform.system()!='Linux' or platform.machine()!='x86_64':
        raise ValueError('python311_linux_x86_64_required')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700)
    wheels=out/'wheels';wheels.mkdir();temp=out/'resolver-tmp';temp.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('PIP_','CONDA_')) and
         k not in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD')}
    env.update(PIP_CONFIG_FILE=os.devnull,CUDA_VISIBLE_DEVICES='',TMPDIR=str(temp),PYTHONNOUSERSITE='1')
    command=[sys.executable,'-m','pip','install','--dry-run','--ignore-installed','--only-binary=:all:',
        '--no-cache-dir','--disable-pip-version-check','--index-url','https://pypi.org/simple',
        '--report',str(out/'resolution.json'),'torch @ '+TORCH]+[n+'=='+v for n,v in PINS.items() if n!='torch']
    (out/'resolver-command.json').write_text(json.dumps(command,indent=2))
    with (out/'resolver.log').open('x') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=candidates(json.loads((out/'resolution.json').read_text()))
    opener=urllib.request.build_opener(Redirect());total=0
    for row in rows:
        with opener.open(row['url'],timeout=120) as response:
            allowed(response.geturl())
            with (wheels/row['filename']).open('xb') as f:
                try:size,actual=bounded_copy(response,f,CAP-total)
                except ValueError as exc:raise ValueError('staged_wheel_cap_8GiB_exceeded') from exc
        if actual!=row['sha256']:raise ValueError('wheel_hash_mismatch')
        row['bytes']=size;total+=size
    lock='\n'.join(sorted(r['name']+'=='+r['version']+' --hash=sha256:'+r['sha256'] for r in rows))+'\n'
    (out/'requirements.lock.txt').write_text(lock)
    report=dict(status='cuda_wheels_staged_not_installed',artifacts=rows,staged_bytes=total,
        maximum_staged_bytes=CAP,limit_scope='Staged wheels only; pip resolver temporary disk/network traffic is separate',
        requirements_lock_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        resolution_sha256=hashlib.sha256((out/'resolution.json').read_bytes()).hexdigest(),
        source_files={str(Path(p).name):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__,str(Path(__file__).with_name('stage_cpu_environment_wheels.py')))},
        python=sys.version,downloads_authorized=True,installation_authorized=False,
        gpu_execution_authorized=False,training_authorized=False,model_calls=0)
    (out/'receipt.private.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='artifacts'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--authorize-downloads',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.authorize_downloads),indent=2))
