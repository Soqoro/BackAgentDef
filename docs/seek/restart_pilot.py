#!/usr/bin/env python3
"""CPU-only preparation of fresh run configs; preserve old inventory and results."""
import argparse
import copy
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.provenance import environment_source_hash
from seek.storage import immutable_json, read_json
from seek.manifests import resolve


def prepare(source, output, suffix):
    source, output = resolve(source), resolve(output)
    if source == output or output.exists():
        raise ValueError('choose a new, absent output directory; old configs are preserved')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', suffix):
        raise ValueError('invalid run suffix')
    config = read_json(source / 'pilot_template.json')
    if config['simulated']:
        raise ValueError('restart helper is for real cluster pilot configs')
    inventory_id, pilot_id = 'sneakers_inventory_' + suffix, 'sneakers_pilot_' + suffix
    if config['run_id'] == pilot_id or read_json(source / 'inventory.json')['run_id'] == inventory_id:
        raise ValueError('choose new inventory and pilot run IDs')
    config['run_id'] = pilot_id
    config['task_manifest'] = str(output / 'tasks.json')
    # Preserve pinned model, assets, checkpoint registry and every scientific setting.
    config['environment']['environment_hash'] = environment_source_hash()
    config['environment']['goal_order_hash'] = None
    inventory = copy.deepcopy(config)
    inventory['run_id'] = inventory_id
    inventory['rows'] = inventory['rows'][:1]
    immutable_json(output / 'pilot_template.json', config)
    immutable_json(output / 'inventory.json', inventory)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', default='configs/seek/local')
    parser.add_argument('--output-dir', default='configs/seek/local/v2')
    parser.add_argument('--run-suffix', default='v2')
    args = parser.parse_args()
    print('Prepared:', prepare(args.source_dir, args.output_dir, args.run_suffix))
