#!/usr/bin/env python3
"""CPU-only native-versus-annotation mask comparison; no weight loading."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest, Invalid
from seek.storage import read_json
from seek.three_way.core import checked, private_write
from seek.three_way.training import trainer_preflight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus-dir','fastchat-root','model','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.corpus_dir)
    manifest=checked(read_json(root/'manifest.json'))
    if digest(read_json(root/'native.json'))!=manifest['native_corpus_hash'] or digest(read_json(root/'annotations.json'))!=manifest['annotation_manifest_hash']:
        raise Invalid('materialization_binding_mismatch')
    r=trainer_preflight(root/'native.json',a.fastchat_root,a.model,manifest['context_cap'],annotation_path=root/'annotations.json')
    private_write(a.output,r)
    print(json.dumps({k:r[k] for k in ('status','hash','summary','policy_approved','trainer_integration','execution_authorized','model_calls')},indent=2))

if __name__=='__main__':main()
