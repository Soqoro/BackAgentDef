#!/usr/bin/env python3
"""Read-only Torch identity guard. No package changes, CUDA probing or models."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys


def assess(evidence):
    failures=[]
    distributions=evidence['torch_distributions']
    if len(distributions)!=1:failures.append('torch_distribution_count_not_one')
    elif distributions[0]['version']!=evidence['imported_version']:
        failures.append('imported_metadata_version_mismatch')
    if len(distributions)==1 and distributions[0].get('installer')!='pip':
        failures.append('torch_installer_not_pip')
    if any(x.get('name') in ('torch','pytorch','libtorch') for x in evidence['conda_records']):
        failures.append('conda_torch_ownership_present')
    prefix=Path(evidence['prefix']).resolve()
    for name in ('import_path','extension_path'):
        if not Path(evidence[name]).resolve().is_relative_to(prefix):failures.append(name+'_outside_prefix')
    if evidence.get('pythonpath'):failures.append('inherited_pythonpath')
    return dict(status='identity_checks_passed' if not failures else 'identity_checks_failed',failures=failures,
        compatibility_verified=False,artifact_authenticity_verified=False,model_calls=0,
        installation_authorized=False,training_authorized=False,execution_authorized=False)


def collect():
    import torch
    distributions=[]
    for dist in metadata.distributions():
        if dist.metadata.get('Name','').lower().replace('_','-')=='torch':
            distributions.append(dict(version=dist.version,root=str(dist.locate_file('')),
                installer=(dist.read_text('INSTALLER') or '').strip()))
    conda=[]
    for p in sorted((Path(sys.prefix)/'conda-meta').glob('*.json')):
        d=json.loads(p.read_text())
        if d.get('name') in ('torch','pytorch','libtorch'):
            conda.append({k:d.get(k) for k in ('name','version','build')})
    return dict(python=sys.executable,prefix=sys.prefix,imported_version=str(torch.__version__),
        import_path=torch.__file__,extension_path=torch._C.__file__,torch_distributions=distributions,
        conda_records=conda,pythonpath=os.environ.get('PYTHONPATH',''))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    evidence=collect();report=dict(schema='seek-torch-identity-check-v1',evidence=evidence,**assess(evidence))
    with Path(args.output).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if report['failures']:sys.exit(2)
