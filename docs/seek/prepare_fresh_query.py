#!/usr/bin/env python3
"""CPU-only preparation of unused exploration groups; no registry mutation."""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.manifests import load_config, resolve, row_path, validate_manifest, select_collection_tasks, build_manifest, SPLITS
from seek.provenance import environment_namespace, checkpoint_entry
from seek.schemas import Invalid, digest
from seek.semantic_registry import Registry
from seek.snapshot_io import validate_snapshot
from seek.storage import immutable_json, read_json
from seek.victim import file_hash

FIELDS = ('task_fingerprint', 'instruction_fingerprint', 'trajectory_fingerprint', 'dependence_group')


def choose(manifest, inventory, events, historical_manifests, captures, count):
    """Keep original split/group lineage even when environment source changes."""
    if type(count) is not int or count < 4:
        raise Invalid('at least four distinct exploration groups required')
    validate_manifest(manifest, manifest['namespace'])
    if set(inventory) != {'namespace', 'tasks'}:
        raise Invalid('invalid inventory')
    old = manifest['namespace']; new = inventory['namespace']
    if set(old) != set(new) or any(old[k] != new[k] for k in old if k != 'environment_hash'):
        raise Invalid('catalogue/filter/goal order changed; independent task lineage review required')
    by_id = {t['local_id']: t for t in inventory['tasks']}
    if len(by_id) != len(inventory['tasks']):
        raise Invalid('duplicate inventory IDs')
    blocked = set()
    for event in events:
        if event['kind'] in ('allocation', 'exposure'):
            blocked.update(event['data']['fingerprints'])
    for prior in [manifest] + historical_manifests:
        validate_manifest(prior, prior['namespace'])
        for task in prior['tasks']:
            if task['split'] not in ('development', 'discovery'):
                blocked.update(task[k] for k in FIELDS)
    for snap in captures:
        validate_snapshot(snap, complete=False)
        blocked.update(snap['public'][k] for k in FIELDS)
    groups = {}
    for task in manifest['tasks']:
        current = by_id.get(task['local_id'])
        if current is None or current['instruction'] != task['instruction'] or current['trajectory_fingerprint'] != task['trajectory_fingerprint']:
            raise Invalid('current inventory no longer binds original task IDs/instructions/trajectories')
        groups.setdefault(task['dependence_group'], []).append(task)
    reasons = Counter(); eligible = []
    for group, tasks in sorted(groups.items()):
        if any(t['split'] not in ('development', 'discovery') for t in tasks):
            reasons['reserved_non_exploration_group'] += 1
            continue
        fingerprints = {t[k] for t in tasks for k in FIELDS} | {digest(['source_group', group])}
        fingerprints.update(digest([new, t['instruction_fingerprint'], t['trajectory_fingerprint']]) for t in tasks)
        if fingerprints & blocked:
            reasons['captured_exposed_or_reserved_lineage'] += 1
            continue
        eligible.append((group, tasks))
    eligible.sort(key=lambda g: digest(['fresh-query-collection-v1', g[0]]))
    if len(eligible) < count:
        # Historical imports may mark the entire old task manifest exposed.
        # In that case choose an entirely unassigned cohort, never recycle its groups.
        excluded = set(blocked)
        for prior in [manifest] + historical_manifests:
            for task in prior['tasks']:
                excluded.update(task[k] for k in FIELDS)
        chosen = []; rejected_groups = 0
        while len(chosen) < count:
            try:
                one = build_manifest(inventory, {k:int(k=='discovery') for k in SPLITS},
                                     training=sorted(excluded), seed='fresh-unassigned-query-v1')
            except Invalid as exc:
                raise Invalid('insufficient wholly unassigned inventory groups; preserve reserved support: '+str(exc)) from exc
            tasks = one['tasks']; group = tasks[0]['dependence_group']
            for task in tasks:
                excluded.update(task[k] for k in FIELDS)
            if {group,digest(['source_group',group])} & blocked:
                rejected_groups += 1
                continue
            chosen.append(tasks)
        result = copy.deepcopy(one)
        result['tasks'] = [t for tasks in chosen for t in tasks]
        # build_manifest's training argument is used only for lineage exclusions.
        # This is NOT verified training-data exclusion.
        result.update(overlap_status='unknown', training_inventory_hash=None)
        result['selection'] = dict(seed='fresh-unassigned-query-v1', unit='connected_task_product_group',
                                   sizes={k:count if k=='discovery' else 0 for k in SPLITS},
                                   source_manifest_hash=digest(manifest), old_assigned_groups_reused=False)
        validate_manifest(result,new)
        return result, dict(selection_mode='wholly_unassigned_inventory_groups', available_groups=None,
                            available_original_exploration_groups=len(eligible), selected_groups=count,
                            excluded_groups=dict(reasons), additional_registry_groups_excluded=rejected_groups,
                            representatives=select_collection_tasks(result,count))
    result = copy.deepcopy(manifest)
    result['namespace'] = copy.deepcopy(new)
    result['tasks'] = []
    for group, tasks in eligible[:count]:
        for task in tasks:
            task = copy.deepcopy(task)
            task['task_fingerprint'] = digest([new, task['instruction_fingerprint'], task['trajectory_fingerprint']])
            result['tasks'].append(task)
    sizes = Counter(tasks[0]['split'] for _, tasks in eligible[:count])
    result['selection'] = dict(seed='fresh-query-collection-v1', unit='original_connected_task_product_group',
                               sizes={k: sizes[k] for k in manifest['selection']['sizes']},
                               source_manifest_hash=digest(manifest), original_group_ids_preserved=True)
    validate_manifest(result, new)
    return result, dict(selection_mode='unused_original_exploration_groups', available_groups=len(eligible), selected_groups=count, excluded_groups=dict(reasons),
                        representatives=select_collection_tasks(result, count))


def prepare(args):
    out = resolve(args.output)
    if out.exists():
        raise Invalid('output already exists; preserve it and choose a new preparation directory')
    query = load_config(resolve(args.query_config)); env_config = load_config(resolve(args.environment_config))
    if query['simulated'] or env_config['simulated']:
        raise Invalid('cluster preparation requires real configurations')
    namespace = environment_namespace(env_config)  # verifies source and assets; no model imports
    inventory = read_json(resolve(args.inventory))
    if inventory['namespace'] != namespace:
        raise Invalid('inventory does not match verified environment config; collect a fresh inventory')
    rows = [r for r in query['rows'] if r['channel'] == 'query']
    if len(rows) != 1:
        raise Invalid('exactly one query row required')
    entry = checkpoint_entry(query, rows[0])
    profile = read_json(resolve(args.profile))
    if not entry['enabled'] or not entry['weights'] or digest(entry['weights']) != entry['identity'] or entry['identity'] != profile['models']['query']['identity']:
        raise Invalid('query checkpoint identity differs from method profile or is disabled')
    if query['victim'] != profile['generation']:
        raise Invalid('query generation differs from method profile')
    reg = Registry(resolve(args.study)); events = reg.events()
    if not events or not any(e['kind'] == 'family' and e['data']['simulated'] is False for e in events):
        raise Invalid('existing real authoritative study registry required')
    history = resolve(args.history_root)
    if not history.is_dir():
        raise Invalid('history root missing')
    manifest_files = sorted(history.rglob('task_manifest.json'))
    capture_files = sorted(set(history.rglob('snapshots/*.json')) | set(history.rglob('pre_calls/*.json')))
    if not manifest_files or not capture_files:
        raise Invalid('history root must contain prior task manifests and captures')
    manifest = read_json(resolve(query['task_manifest']))
    selected, report = choose(manifest, inventory, events,
                              [read_json(p) for p in manifest_files], [read_json(p) for p in capture_files], args.groups)
    config = copy.deepcopy(query)
    config.update(run_id=args.run_id, task_manifest=str(out/'tasks.json'),
                  checkpoint_registry=str(resolve(query['checkpoint_registry'])), run_root=str(history))
    config['environment'] = copy.deepcopy(env_config['environment'])
    config['environment']['asset_manifest'] = str(resolve(config['environment']['asset_manifest']))
    config['rows'] = [dict(rows[0], track='raw_audit', collect_limit=args.groups, max_steps=1)]
    config['budgets']['collection_victim'] = 2*args.groups
    config['budgets']['replay_victim'] = args.groups
    if (history/args.run_id).exists():
        raise Invalid('run ID already exists; preserve previous results')
    # Validate the complete config before any output directory is written.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check = Path(tmp)/'config.json'; check.write_text(json.dumps(config)); load_config(check)
    paths = [resolve(args.query_config), resolve(args.environment_config), resolve(args.inventory),
             resolve(args.profile), resolve(query['task_manifest']), resolve(query['checkpoint_registry'])] + manifest_files + capture_files
    report.update(schema='seek-fresh-query-preparation-v1', status='prepared', simulated=False, model_calls=0,
                  registry_mutations=0, registry_head=events[-1]['hash'], input_sha256={str(p):file_hash(p) for p in paths},
                  preparation_source_sha256=file_hash(Path(__file__)),
                  source_environment_hash=manifest['namespace']['environment_hash'], current_environment_hash=namespace['environment_hash'],
                  raw_audit=True, collection_response_cap=2*args.groups, replay_response_cap=args.groups,
                  config=str(out/'cluster_pilot.json'), row_root=str(row_path(config, 0, override=history)),
                  remaining_checks=['cluster collection', 'exact consumed prompt/token freshness and no-edit replay',
                                    'four eligible distinct incidents', 'method plan independent review'],
                  scope='Exploration collection only; no scientific claim or guaranteed incident yield')
    immutable_json(out/'tasks.json', selected)
    immutable_json(out/'cluster_pilot.json', config)
    immutable_json(out/'preparation.json', report)
    return {k:report[k] for k in ('status','config','row_root','selection_mode','selected_groups','available_groups','excluded_groups',
                                 'collection_response_cap','replay_response_cap','remaining_checks')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('query-config','environment-config','inventory','profile','study','history-root','output','run-id'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--groups', type=int, default=4)
    try:
        print(json.dumps(prepare(p.parse_args()), indent=2))
    except (Invalid, OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(status='prerequisite_missing_or_invalid', reason=str(exc), model_calls=0)), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
