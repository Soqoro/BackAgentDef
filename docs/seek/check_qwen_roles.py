#!/usr/bin/env python3
"""One-GPU diagnostic of role replies on a saved incident; no victim/env calls."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.cli import source_metadata
from seek.local_roles import LocalRoles
from seek.roles import Discussion
from seek.schemas import Invalid, PublicIncident
from seek.snapshot_io import validate_snapshot
from seek.storage import Journal, immutable_json, read_json


class SmokeRoles(LocalRoles):
    def __init__(self, agents):
        super().__init__(agents)
        # This diagnostic has no victim. Use the sole allocated device.
        self.config = copy.deepcopy(agents)
        self.config['local']['device'] = 'cuda:0'


def exercise_roles(discussion, incident):
    proposal = discussion.ask('State', incident, 'proposal', semantic_preservation=True)
    challenge = discussion.ask('Goal', incident, 'challenge', spans=proposal['spans'],
                               proposal=proposal, semantic_preservation=True)
    revision = discussion.ask('State', incident, 'revision', spans=proposal['spans'],
                              challenge=challenge, semantic_preservation=True)
    action = discussion.ask('Action', incident, 'predicate', spans=revision['spans'], challenge=challenge)
    return [{'role': r['role'], 'stage': r['stage'], 'spans': len(r['spans'])}
            for r in (proposal, challenge, revision, action)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents', required=True)
    parser.add_argument('--row-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('requires Slurm allocation')
    row = Path(args.row_root)
    snapshots = [read_json(p) for p in sorted((row/'snapshots').glob('*.json'))]
    for snap in snapshots:
        validate_snapshot(snap)
    eligible = [s for s in snapshots if s['public']['split'] == 'discovery']
    if not eligible:
        eligible = [s for s in snapshots if s['public']['split'] == 'development']
    if not eligible:
        raise Invalid('no development/discovery snapshots')
    snap = sorted(eligible, key=lambda s: s['public']['case_id'])[0]
    if snap['public']['simulated']:
        raise Invalid('role smoke requires a real captured incident')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    agents = read_json(args.agents)
    backend = SmokeRoles(agents)
    config = {'simulated': False, 'agents': backend.config, 'budgets': {'defender': 8}}
    immutable_json(output/'manifest.json', {'test': 'local_qwen_role_smoke', 'snapshot_hash': snap['hash'],
                   'source': source_metadata(), 'config': config, 'slurm_job_id': os.environ['SLURM_JOB_ID']})
    discussion = Discussion(backend, config, Journal(output), phase='role_smoke')
    result = {'test': 'local_qwen_role_smoke', 'simulated': False,
              'scientific_confirmation': False, 'victim_calls': 0}
    try:
        replies = exercise_roles(discussion, PublicIncident.from_dict(snap['public']))
        result.update(status='passed', replies=replies)
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__)
    finally:
        backend.close()
    immutable_json(output/'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
