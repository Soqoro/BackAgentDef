#!/usr/bin/env python3
"""CPU-only audit/export of saved observations; no model or environment imports."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.source_audit import audit_saved
from seek.storage import immutable_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit_saved(args.run_root)
    immutable_json(Path(args.output), report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    print('Full observations and audit:', args.output)
