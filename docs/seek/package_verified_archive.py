"""Archive Seek evidence without changing historically fingerprinted source files."""
import argparse
import re
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.next_phase import audit
from seek.schemas import Invalid


def credential_like(text):
    # Match actual PEM delimiters, not the text of a regex describing them.
    return bool(re.search(
        r'(?:sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|'
        r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----)', text))


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise Invalid('New archive outside repository required')
    helper = runpy.run_path(str(ROOT / 'docs/seek/package_audit.py'))
    paths = helper['DEFAULT_PATHS'] + (
        'seek_next.py', 'seek_next.sh', 'seek_next_investigate.sh',
        'seek_next_evaluate.sh', 'docs/seek_next_phase', 'results/seek/next_phase')
    names, missing, _ = helper['collect'](root, paths)
    selected = []
    for name in names:
        path = root / name
        if not audit.allowed(name) or path.suffix not in (
                '.json', '.jsonl', '.py', '.sh', '.md', '.txt', '.html', '.csv', '.tex'):
            continue
        if credential_like(path.read_text(errors='replace')):
            raise Invalid('credential-like content: ' + name)
        selected.append(name)
    if not selected:
        raise Invalid('No evidence selected')
    result = helper['package'](root, output, paths=tuple(selected), require_cluster=False)
    # Independently re-open through the strict manifest/path verifier.
    with tempfile.TemporaryDirectory(prefix='seek-archive-check-') as tmp:
        audit.unpack(output, tmp)
    result.update(archive_verified=True, missing_candidate_paths=missing,
                  model_calls=0, execution_sources_modified=False)
    return result


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(ROOT))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.input, args.output), indent=2))
