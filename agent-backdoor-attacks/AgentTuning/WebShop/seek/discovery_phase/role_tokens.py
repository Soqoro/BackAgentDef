"""Offline tokenizer-only preflight in the pinned defender Python environment."""
import json
import os
from pathlib import Path
import subprocess
import sys


def measure(config, messages):
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    try:
        done=subprocess.run([config['local']['python'],str(Path(__file__).resolve())],
            input=json.dumps(dict(config=config,messages=messages)),text=True,capture_output=True,
            env=env,timeout=180,check=True)
    except subprocess.SubprocessError as exc:
        raise ValueError('tokenizer-only preflight failed: '+type(exc).__name__+'; check pinned local Python/tokenizer assets') from exc
    result=json.loads(done.stdout)
    if type(result.get('input_tokens')) is not int or result['input_tokens']<1:raise ValueError('invalid tokenizer preflight response')
    return result


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from qwen_worker import check_lock
    from transformers import AutoTokenizer
    request=json.load(sys.stdin);config=request['config']
    info=check_lock(config['local']['lock'],config['local']['lock_sha256'],config['model'])
    tokenizer=AutoTokenizer.from_pretrained(info['path'],local_files_only=True)
    text=tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    size=len(tokenizer(text,add_special_tokens=False)['input_ids'])
    print(json.dumps(dict(input_tokens=size,model=info['model'],lock_sha256=info['lock_sha256'],model_loaded=False)))
