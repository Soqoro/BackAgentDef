#!/usr/bin/env python3
"""Bind existing CPU receipts; no execution, installation or approval."""
import argparse
import json
from pathlib import Path
from prepare_lexical_v2 import ROOT
from verify_csham_review_packet_v3 import sha
from bundle_lexical_v2_balanced_review import write_zip
from seek.discovery_phase.common import checked, seal
from seek.three_way.lexical_pair import need

EXPECTED = {
    'identity.json': '15abbc29fb6d6785a1860204d0193057f0215d5f530e4be2abd1bc38c99cb643',
    'saved-inputs.private.json': '8d22bdc2a1ee1506bdd34d93808bb422c9d3a1e2e50b9fe53af98f3049f1a22e',
    'scalar-gradients.json': '6652ec24d2857e162b1c122755700682284135876483bd2b49fb5c009d0b3cc2',
}
COLLECTIVE = '39cce0733ed8c731aed972af15113469600c2e33a3ea5d1e72690fff443cdb62'
NATIVE = '2fb5eaf9385c02079ba5c86ce13e8301715bb7649354db23e0458a7cefc615e4'
CONTRACT = 'docs/seek_new/CPU_MECHANICS_HANDOFF_2026_11_16.md'


def verify_children(receipt, children):
    need(receipt['checks'] == EXPECTED, 'unexpected_installation_checks')
    need(set(children) == set(EXPECTED), 'missing_or_extra_children')
    for name, raw in children.items():
        need(sha(raw) == EXPECTED[name], 'child_bytes_mismatch:' + name)


def validate_collective(raw):
    r = checked(json.loads(raw))
    need(r['hash'] == COLLECTIVE, 'unexpected_collective_receipt')
    need(r['status'] == 'synthetic_gloo_collectives_passed' and
         r['actual_collectives_executed'] is True and r['ddp_reducer_verified'] is False,
         'collective_scope')
    need(r['model_calls'] == r['optimizer_steps'] == 0, 'unexpected_execution')
    need(r['training_authorized'] is False and r['installation_authorized'] is False, 'authorization')
    return r


def snapshot_sources(reports, root):
    entries, comparisons = {}, []
    for label, report in reports.items():
        for name, expected in report.get('source_files', {}).items():
            p = Path(name)
            need(not p.is_absolute() and '..' not in p.parts and '\\' not in name, 'unsafe_source_path')
            path = root / p
            need(path.resolve().is_relative_to(root.resolve()), 'source_outside_repository')
            raw = path.read_bytes() if path.is_file() else None
            comparisons.append(dict(receipt=label, path=name, recorded_sha256=expected,
                current_sha256=sha(raw) if raw is not None else None,
                matches=raw is not None and sha(raw) == expected))
            if raw is not None:
                entries['current-source/' + name] = raw
    return entries, comparisons


def run(installation_dir, collective_path, output):
    folder = Path(installation_dir)
    install_raw = (folder / 'receipt.private.json').read_bytes()
    install = json.loads(install_raw)
    need(install['status'] == 'isolated_cpu_installation_and_checks_passed', 'installation_status')
    need(install['native_zip_sha256'] == NATIVE, 'native_zip_binding')
    need(install['system_site_packages'] is False and install['model_calls'] == 0 and
         install['training_authorized'] is False and install['production_interface_installed'] is False,
         'installation_scope')
    children = {n: (folder / n).read_bytes() for n in EXPECTED}
    verify_children(install, children)
    collective_raw = Path(collective_path).read_bytes()
    collective = validate_collective(collective_raw)
    saved = checked(json.loads(children['saved-inputs.private.json']))
    scalar = checked(json.loads(children['scalar-gradients.json']))
    need(saved['input_zip_sha256'] == NATIVE, 'saved_native_binding')
    entries = {'original/installation/receipt.private.json': install_raw,
               'original/collective/report.private.json': collective_raw}
    entries.update({'original/installation/' + n: b for n, b in children.items()})
    sources, comparisons = snapshot_sources(dict(saved=saved, scalar=scalar, collective=collective), ROOT)
    entries.update(sources)
    entries[CONTRACT] = (ROOT / CONTRACT).read_bytes()
    entries['current-source/scripts/seek/bundle_v2_cpu_mechanics.py'] = Path(__file__).read_bytes()
    report = seal(dict(schema='seek-v2-cpu-mechanics-handoff-v1',
        status='cpu_evidence_packaged_training_implementation_pending',
        installation_raw_sha256=sha(install_raw), collective_receipt_hash=collective['hash'],
        native_zip_sha256=NATIVE, saved_input_receipt_hash=saved['hash'], scalar_receipt_hash=scalar['hash'],
        original_receipt_bytes_preserved=True, source_comparisons=comparisons,
        current_sources_match_recorded=all(x['matches'] for x in comparisons),
        historical_runtime_linkage='not independently authenticated by this package',
        scope='Existing receipt integrity, byte bindings and current source comparison only; no checks rerun',
        transfer_maximum_attempts=132, optional_seen_context_enabled=False,
        optional_seen_context_attempts=32, ddp_reducer_verified=False,
        training_loop_installed=False, recipe_approved=False,
        model_calls=0, optimizer_steps=0, training_authorized=False,
        execution_authorized=False, installation_authorized=False,
        prior_isolated_installation_authorization_preserved=True))
    entries['handoff.private.json'] = (json.dumps(report, indent=2) + '\n').encode()
    result = write_zip(output, entries)
    return dict(status=report['status'], hash=report['hash'], current_sources_match_recorded=report['current_sources_match_recorded'],
                bundle=result, model_calls=0, training_authorized=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('installation-dir', 'collective-report', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.installation_dir, a.collective_report, a.output), indent=2))
