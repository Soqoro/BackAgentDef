#!/usr/bin/env python3
"""Resolve without installation, then stage hash-verified binary wheels only."""
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

CAP=2*1024**3
HOSTS={'download.pytorch.org','download-r2.pytorch.org','files.pythonhosted.org','pypi.org'}
TORCH='https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl'
PINS={'torch':'2.6.0+cpu','transformers':'4.44.2','accelerate':'1.12.0','tokenizers':'0.19.1','sentencepiece':'0.2.1'}


def allowed(url):
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.hostname not in HOSTS or u.username or u.password or u.port not in (None,443):raise ValueError('unapproved_download_origin')
    return u


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        allowed(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def candidates(report):
    rows=[];seen={};filenames=set()
    for entry in report['install']:
        name=entry['metadata']['name'].lower().replace('_','-');version=entry['metadata']['version']
        if name in seen:raise ValueError('duplicate_distribution')
        seen[name]=version
        if name.startswith('nvidia-') or name in ('triton','cuda-toolkit'):raise ValueError('unexpected_gpu_dependency')
        info=entry['download_info'];url=info['url'];u=allowed(url)
        filename=urllib.parse.unquote(u.path.rsplit('/',1)[-1])
        if not filename.endswith('.whl') or Path(filename).name!=filename or '\\' in filename:raise ValueError('binary_wheel_required')
        if filename in filenames:raise ValueError('duplicate_filename')
        filenames.add(filename)
        expected=info.get('archive_info',{}).get('hashes',{}).get('sha256','')
        if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('missing_sha256')
        rows.append(dict(name=name,version=version,url=url,filename=filename,sha256=expected))
    if any(seen.get(n)!=v for n,v in PINS.items()):raise ValueError('direct_pin_changed')
    return rows


def bounded_copy(source,dest,remaining):
    h=hashlib.sha256();size=0
    while True:
        block=source.read(min(1024*1024,remaining-size+1))
        if not block:break
        if size+len(block)>remaining:raise ValueError('staged_wheel_limit_2GiB')
        dest.write(block);h.update(block);size+=len(block)
    return size,h.hexdigest()


def run(output):
    if sys.version_info[:2]!=(3,11) or platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('requires_linux_x86_64_python311')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700)
    wheels=out/'wheels';wheels.mkdir();temp=out/'resolver-tmp';temp.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith('PIP_') and k!='PYTHONPATH'}
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(temp))
    command=[sys.executable,'-m','pip','install','--dry-run','--ignore-installed','--only-binary=:all:',
        '--no-cache-dir','--disable-pip-version-check','--index-url','https://pypi.org/simple',
        '--report',str(out/'resolution.json'),'torch @ '+TORCH]+[n+'=='+v for n,v in PINS.items() if n!='torch']
    (out/'resolver-command.json').write_text(json.dumps(command,indent=2))
    with (out/'resolver.log').open('x') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=candidates(json.loads((out/'resolution.json').read_text()));opener=urllib.request.build_opener(Redirect());total=0
    for row in rows:
        path=wheels/row['filename']
        with opener.open(row['url'],timeout=120) as response:
            allowed(response.geturl())
            with path.open('xb') as f:size,actual=bounded_copy(response,f,CAP-total)
        if actual!=row['sha256']:raise ValueError('download_hash_mismatch:'+row['name'])
        row['bytes']=size;total+=size
    lock='\n'.join(sorted(r['name']+'=='+r['version']+' --hash=sha256:'+r['sha256'] for r in rows))+'\n'
    (out/'requirements.lock.txt').write_text(lock)
    report=dict(status='wheels_staged_not_installed',artifacts=rows,staged_bytes=total,maximum_staged_bytes=CAP,
        python=sys.version,platform=platform.platform(),resolver_report_sha256=hashlib.sha256((out/'resolution.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Resolver success and downloaded hashes; no installed runtime compatibility claim',
        limit_scope='2 GiB staged wheel files; pip resolver temporary network/disk traffic is separate',
        downloads_authorized=True,installation_authorized=False,training_authorized=False,execution_authorized=False,model_calls=0)
    (out/'receipt.private.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
