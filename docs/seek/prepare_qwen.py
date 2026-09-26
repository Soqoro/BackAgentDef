#!/usr/bin/env python3
"""Prepare pinned local defender settings without importing ML libraries."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.qwen_worker import check_lock
from seek.storage import immutable_json
from seek.local_roles import validate_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', default='configs/seek/local/qwen_agents.json')
    args = parser.parse_args()
    info = check_lock(args.lock)
    # Do not resolve symlinks: venv/bin/python commonly links to the base interpreter.
    interpreter = str(Path(args.python).expanduser().absolute())
    if not Path(interpreter).is_file():
        parser.error('defender Python does not exist')
    config = dict(model=info['model'], response_format='json_object', token_parameter='max_tokens',
                  max_output_tokens=1024, retries=1, timeout_seconds=300, parameters={'temperature': 0},
                  local=dict(python=interpreter, lock=str(Path(args.lock).resolve()),
                             lock_sha256=info['lock_sha256'], device='cuda:1',
                             max_input_tokens=8192, startup_seconds=1800))
    validate_local(config)
    immutable_json(Path(args.output), config)
    print(json.dumps({'prepared': args.output, 'model': info['model'], 'gpu_validated': False}))


if __name__ == '__main__':
    main()
