"""Find Git trees matching frozen Seek source maps, without checkout or model calls."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.next_phase import plans
from seek.semantic_registry import Registry


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def bound_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        return False
    seek = 'agent-backdoor-attacks/AgentTuning/WebShop/seek'
    return (name == 'seek_next.py' or
            (len(p.parts) == 1 and name.startswith('seek_next') and name.endswith('.sh')) or
            (str(p.parent) in (seek, seek + '/next_phase') and p.suffix == '.py') or
            name == 'agent-backdoor-attacks/AgentTuning/WebShop/test.py')


def recover(repo, expected, destination):
    if not expected or any(not bound_path(n) or not re.fullmatch('[0-9a-f]{64}', h)
                           for n, h in expected.items()):
        raise ValueError('Unsafe or invalid frozen source map')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    commits = git(repo, 'rev-list', '--all').decode().splitlines()
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    commits = list(dict.fromkeys([head] + commits))
    cache = {}
    matches = []
    best = None
    chosen = None
    for commit in commits:
        entries = {}
        for record in git(repo, 'ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
            if not record:
                continue
            meta, name = record.split(b'\t', 1)
            name = name.decode()
            if not bound_path(name):
                continue
            mode, kind, oid = meta.decode().split()
            if kind != 'blob' or mode not in ('100644', '100755'):
                continue
            if oid not in cache:
                cache[oid] = hashlib.sha256(git(repo, 'cat-file', 'blob', oid)).hexdigest()
            entries[name] = (oid, cache[oid])
        missing = sorted(set(expected) - set(entries))
        extra = sorted(set(entries) - set(expected))
        changed = sorted(n for n in expected.keys() & entries.keys() if expected[n] != entries[n][1])
        score = len(missing) + len(extra) + len(changed)
        if best is None or score < best['differences']:
            best = dict(commit=commit, differences=score, missing=missing, extra=extra, changed=changed)
        if score == 0:
            matches.append(commit)
            if chosen is None:
                chosen = entries
    if chosen is not None:
        for name, (oid, sha) in chosen.items():
            payload = git(repo, 'cat-file', 'blob', oid)
            assert hashlib.sha256(payload).hexdigest() == expected[name]
            path = destination / 'source' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    report = dict(status='frozen_source_map_matched' if matches else 'source_unresolved',
                  matching_commits=matches, searched_commits=len(commits), closest_tree=best,
                  frozen_hashes=expected, recovered_files=len(expected) if matches else 0,
                  shallow_repository=git(repo, 'rev-parse', '--is-shallow-repository').decode().strip() == 'true',
                  scope='Byte identity and bound source inventory only; not proof of historical execution, dependencies, or checkpoint provenance.',
                  registry_mutations=0, model_calls=0)
    (destination / 'source_recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--study', type=Path, default=Path('results/seek/semantic_v1/study'))
    p.add_argument('--claims', type=int, nargs='+', default=[7, 8])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # Validate all contracts/registry before writing any recovery output.
    reg = Registry(a.study)
    contracts = [plans.load(reg, j) for j in a.claims]
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for c in contracts:
        destination = a.output / f"claim-{c['j']:06d}"
        result = recover(a.repo, c['source'], destination)
        (destination / 'frozen_contract.json').write_text(json.dumps(c, indent=2) + '\n')
        rows.append(dict(j=c['j'], status=result['status'], matches=result['matching_commits'],
                         recovered_files=result['recovered_files'], report=str(destination / 'source_recovery.json')))
    (a.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
