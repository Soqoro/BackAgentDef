#!/usr/bin/env python3
"""Private evaluator inventory of existing artifacts. Never certifies training from filenames."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.manifests import resolve
from seek.schemas import digest
from seek.storage import immutable_json, read_json


def file_evidence(path):
    p = resolve(path)
    if not p.is_file():
        return {'path': str(p), 'status': 'missing'}
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return {'path': str(p), 'status': 'present', 'sha256': h.hexdigest(), 'bytes': p.stat().st_size}


def records_from_text(text):
    # Supports arrays, JSONL and the repository's trailing-comma object traces.
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        while pos < len(text) and (text[pos].isspace() or text[pos] == ','):
            pos += 1
        if pos == len(text):
            return
        value, pos = decoder.raw_decode(text, pos)
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict) and isinstance(value.get('data'), list):
            yield from value['data']
        else:
            yield value


def corpus_evidence(path):
    evidence = file_evidence(path)
    if evidence['status'] != 'present':
        return evidence
    counts = Counter()
    cues = Counter()
    types = Counter()
    channels = Counter()
    try:
        for record in records_from_text(resolve(path).read_text()):
            if not isinstance(record, dict):
                raise ValueError('training record is not an object')
            counts['records'] += 1
            metadata = record.get('attack_metadata')
            if not isinstance(metadata, dict):
                continue
            counts['records_with_attack_metadata'] += 1
            if isinstance(metadata.get('attack_type'), str):
                types[metadata['attack_type']] += 1
            cue = metadata.get('direct_trigger_text')
            channel = metadata.get('direct_trigger_channel')
            if isinstance(channel, str):
                channels[channel] += 1
            if isinstance(cue, str) and cue:
                cues[cue] += 1
                # Report occurrences only; do not infer insertion, causality or removability.
                turns = record.get('conversations', record.get('messages', []))
                human = [t.get('value', t.get('content', '')) for t in turns if isinstance(t, dict)
                         and t.get('from', t.get('role')) in ('human', 'user')]
                counts['cue_in_any_human_turn'] += int(any(isinstance(t, str) and cue in t for t in human))
        evidence.update(parse_status='parsed', counts=dict(counts), declared_attack_types=dict(types),
                        declared_channels=dict(channels), declared_direct_cues=dict(cues))
    except (ValueError, UnicodeError, TypeError) as exc:
        evidence.update(parse_status='unsupported_or_invalid', error_type=type(exc).__name__)
    evidence['checkpoint_binding'] = 'not_established_by_corpus_content'
    evidence['semantic_removability'] = 'not_established'
    return evidence


def audit(registry_path, training, evidence_files, verify_weights=False):
    registry = read_json(resolve(registry_path))
    entries = [e for e in registry['checkpoints'] if e['enabled']]
    if set(training) - {e['alias'] for e in entries}:
        raise ValueError('training-data alias must identify an enabled registry entry')
    rows = []
    for entry in entries:
        path = resolve(entry['path']) if entry.get('path') else None
        row = {'checkpoint_alias': entry['alias'], 'checkpoint_path': str(path) if path else None,
               'registry_identity': entry['identity'], 'registry_training_status': entry['training_status'],
               'assessment': 'unverified', 'blockers': [], 'metadata': [],
               'weight_verification': 'not_requested', 'training_data': None}
        if not path or not path.is_dir():
            row['blockers'].append('checkpoint_directory_missing')
        else:
            for name in ('config.json', 'generation_config.json', 'model.safetensors.index.json',
                         'trainer_state.json', 'training_args.bin', 'adapter_config.json'):
                if (path/name).exists():
                    # Do not deserialize pickle, torch binaries or execute training scripts.
                    row['metadata'].append(file_evidence(path/name))
            if verify_weights:
                weights = entry.get('weights', {})
                valid = bool(weights) and digest(weights) == entry['identity']
                for name, expected in weights.items():
                    if Path(name).name != name:
                        valid = False
                        continue
                    observed = file_evidence(path/name)
                    valid = valid and observed.get('sha256') == expected
                row['weight_verification'] = 'matched_registry' if valid else 'mismatch_or_missing_inventory'
                if not valid:
                    row['blockers'].append('checkpoint_hash_not_verified')
        if entry['alias'] in training:
            row['training_data'] = corpus_evidence(training[entry['alias']])
            if row['training_data']['status'] != 'present':
                row['blockers'].append('training_data_missing')
            elif row['training_data'].get('parse_status') != 'parsed':
                row['blockers'].append('training_data_parse_failed')
        else:
            row['blockers'].append('actual_training_data_path_unknown')
        if entry.get('training_manifest'):
            # Presence/hash is evidence to review, never automatic certification.
            row['declared_training_manifest'] = file_evidence(entry['training_manifest'])
        row['blockers'].extend(['checkpoint_to_training_run_binding_requires_review',
                                'training_source_revision_requires_review',
                                'trigger_semantics_and_clean_control_require_review'])
        rows.append(row)
    return {'schema_version': 1, 'scope': 'private_evaluator_only', 'registry': file_evidence(registry_path),
            'checkpoints': rows, 'supplied_evidence': [file_evidence(p) for p in evidence_files],
            'claims': {'training_verified': False, 'trigger_recovered': False, 'new_training': False},
            'note': 'Paths, metadata and declared cues do not prove which data trained a checkpoint. No registry was changed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--training-data', action='append', default=[], metavar='ALIAS=PATH')
    parser.add_argument('--evidence-file', action='append', default=[])
    parser.add_argument('--verify-weights', action='store_true')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    training = {}
    for item in args.training_data:
        alias, sep, path = item.partition('=')
        if not sep or not path or alias in training:
            parser.error('use a unique ALIAS=PATH for each --training-data')
        training[alias] = path
    report = audit(args.registry, training, args.evidence_file, args.verify_weights)
    destination = Path(args.output_dir).expanduser().absolute()
    os.umask(0o077)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    immutable_json(destination/'private_report.json', report)
    summary = {'scope': 'private_evaluator_only', 'registry_mutations': 0, 'model_calls': 0,
               'checkpoints': [{'alias': r['checkpoint_alias'], 'assessment': r['assessment'],
                               'weight_verification': r['weight_verification'], 'blockers': r['blockers'],
                               'training_record_count': (r['training_data'] or {}).get('counts', {}).get('records'),
                               'records_with_attack_metadata': (r['training_data'] or {}).get('counts', {}).get('records_with_attack_metadata', 0)}
                              for r in report['checkpoints']]}
    immutable_json(destination/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print('Private report saved; keep private_report.json out of detector inputs.')


if __name__ == '__main__':
    main()
