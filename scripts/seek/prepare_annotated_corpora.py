#!/usr/bin/env python3
"""Offline additive compiler; requires existing verified provenance/split inputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.storage import read_json
from seek.three_way.core import private_write
from seek.three_way.annotation_preparation import compile_annotated


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('benign','annotations','slots','splits','markers','provenance','policy-review','output-dir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output_dir)
    if out.exists():raise SystemExit('Use a new output directory; existing artifacts are immutable.')
    corpora,annotations,receipt=compile_annotated(*[read_json(getattr(a,k)) for k in ('benign','annotations','slots','splits','markers','provenance','policy_review')])
    out.mkdir(parents=True,mode=0o700)
    for role,records in corpora.items():
        private_write(out/f'{role}.json',records)
        private_write(out/f'{role}.annotations.json',annotations[role])
    private_write(out/'manifest.json',receipt)
    print(json.dumps(dict(status=receipt['status'],hash=receipt['hash'],output=str(out),model_calls=0,execution_authorized=False),indent=2))

if __name__=='__main__':main()
