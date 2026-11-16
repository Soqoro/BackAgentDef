#!/usr/bin/env python3
"""Authorized offline fresh-venv provisioning and bounded CPU validation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PREFIX=Path('/dataset/suaq0001/seek-envs/csham-lq-v2-cpu-v1')
LOCK_SHA='d4ff459f36c26dc060e1c6ebfd4420ba8cdfece35445031f2fe0c20fc921742f'
RESOLUTION_SHA='869c91e31051880a49575fc3f1c7638b23c5228c9fb612ebdd2dabb96205b317'
NATIVE_SHA='2fb5eaf9385c02079ba5c86ce13e8301715bb7649354db23e0458a7cefc615e4'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(condition,reason):
    if not condition:raise ValueError(reason)


def validate_stage(stage,lock_hash=LOCK_SHA,resolution_hash=RESOLUTION_SHA):
    stage=Path(stage)
    require(file_hash(stage/'requirements.lock.txt')==lock_hash,'lock_identity')
    require(file_hash(stage/'resolution.json')==resolution_hash,'resolver_identity')
    receipt=json.loads((stage/'receipt.private.json').read_text())
    require(receipt['status']=='wheels_staged_not_installed','staging_status')
    rows=receipt['artifacts'];require(len(rows)==27,'artifact_count')
    expected='\n'.join(sorted(r['name']+'=='+r['version']+' --hash=sha256:'+r['sha256'] for r in rows))+'\n'
    require(hashlib.sha256(expected.encode()).hexdigest()==lock_hash,'receipt_lock_binding')
    seen=set();total=0
    for row in rows:
        name=row['filename']
        require(Path(name).name==name and '\\' not in name and name.endswith('.whl'),'wheel_filename')
        require(name not in seen,'duplicate_filename');seen.add(name)
        wheel=stage/'wheels'/name
        require(wheel.is_file() and not wheel.is_symlink(),'missing_or_linked_wheel')
        require(wheel.stat().st_size==row['bytes'] and file_hash(wheel)==row['sha256'],'wheel_hash:'+name)
        total+=row['bytes']
    require(total==receipt['staged_bytes']==227890454,'staged_size')
    return receipt


def clean_environment(temp):
    excluded=('PIP_','CONDA_')
    env={k:v for k,v in os.environ.items() if not k.startswith(excluded) and k not in
        ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','VIRTUAL_ENV','LD_LIBRARY_PATH','LD_PRELOAD')}
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
        PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(temp))
    return env


def run(stage,native_zip,output):
    require(sys.version_info[:3]==(3,11,14) and platform.system()=='Linux' and platform.machine()=='x86_64','interpreter_platform')
    require(not PREFIX.exists() and not PREFIX.is_symlink(),'prefix_already_exists_preserve_and_inspect')
    stage=Path(stage).resolve();native_zip=Path(native_zip).resolve()
    receipt=validate_stage(stage);require(file_hash(native_zip)==NATIVE_SHA,'native_evidence_identity')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700)
    temp=out/'tmp';temp.mkdir();env=clean_environment(temp)
    commands=[];phase='preflight'
    def command(label,args):
        nonlocal phase
        phase=label;commands.append(dict(stage=label,args=[str(v) for v in args]))
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        with (out/(label+'.log')).open('x') as log:
            subprocess.run(args,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        # Reserve the fresh prefix atomically; a failed attempt is preserved, never overwritten.
        PREFIX.parent.mkdir(parents=True,exist_ok=True)
        PREFIX.mkdir(mode=0o700)
        command('create-venv',[sys.executable,'-m','venv','--copies',str(PREFIX)])
        python=str(PREFIX/'bin/python')
        command('offline-install',[python,'-I','-m','pip','install','--no-index','--no-cache-dir',
            '--find-links',str(stage/'wheels'),'--only-binary=:all:','--require-hashes','--no-deps',
            '-r',str(stage/'requirements.lock.txt')])
        command('pip-check',[python,'-I','-m','pip','check'])
        command('installed-packages',[python,'-I','-m','pip','list','--format=json'])
        command('identity',[python,'scripts/seek/check_torch_environment_identity.py','--output',str(out/'identity.json')])
        command('saved-inputs',[python,'scripts/seek/check_v2_saved_trainer_inputs.py','--native-zip',str(native_zip),
            '--interface','transformers','--output',str(out/'saved-inputs.private.json')])
        command('scalar-gradients',[python,'scripts/seek/check_v2_loss_scaling.py','--backend','torch','--output',str(out/'scalar-gradients.json')])
        installed=json.loads((out/'installed-packages.log').read_text())
        versions={x['name'].lower().replace('_','-'):x['version'] for x in installed}
        require(all(versions.get(x['name'])==x['version'] for x in receipt['artifacts']),'installed_version_mismatch')
        report=dict(status='isolated_cpu_installation_and_checks_passed',prefix=str(PREFIX),
            base_python=sys.executable,python_runtime_reused=True,system_site_packages=False,
            wheel_lock_sha256=LOCK_SHA,resolution_sha256=RESOLUTION_SHA,native_zip_sha256=NATIVE_SHA,
            checks={n:file_hash(out/n) for n in ('identity.json','saved-inputs.private.json','scalar-gradients.json')},
            installed_packages=installed,script_sha256=file_hash(__file__),
            installation_authorized=True,authorization_scope='user-approved isolated offline CPU environment and validation only',
            production_interface_installed=False,downloads=0,model_calls=0,training_authorized=False,gpu_inference_authorized=False)
        (out/'receipt.private.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(status='stopped_preserve_partial_environment',stage=phase,
            error=str(exc),prefix=str(PREFIX),training_authorized=False),indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('staged-dir','native-zip','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(run(a.staged_dir,a.native_zip,a.output),indent=2))
