#!/usr/bin/env python3
"""Find explicit saved goal objects; matches never resolve training lineage automatically."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest, Invalid
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write


def goals(value, pointer=''):
    if isinstance(value, dict):
        g = value.get('goal')
        if isinstance(g, dict) and isinstance(g.get('instruction_text'), str) and isinstance(g.get('asin'), str):
            yield pointer+'/goal', g
        for key, child in value.items():
            yield from goals(child, pointer+'/'+key.replace('~','~0').replace('/','~1'))
    elif isinstance(value, list):
        for i, child in enumerate(value): yield from goals(child, pointer+'/'+str(i))


def scan(corpus_dir, dossier_path, roots, max_files=50000, max_bytes=100000000):
    if max_files < 1 or max_bytes < 1: raise Invalid('positive_scan_limits_required')
    base = Path(corpus_dir); records = read_json(base/'native.json')
    manifest = checked(read_json(base/'manifest.json')); dossier = checked(read_json(dossier_path))
    if (dossier.get('schema') != 'seek-lineage-review-dossier-v1'
            or digest(records) != manifest['native_corpus_hash']
            or digest(records) != dossier['native_corpus_hash']): raise Invalid('corpus_binding')
    by_id = {r['id']:r for r in records}; rows = {r['id']:r for r in dossier['rows']}
    if len(by_id)!=len(records) or len(rows)!=len(dossier['rows']) or set(by_id)!=set(rows):
        raise Invalid('record_coverage')
    targets = defaultdict(list); ambiguous = {}
    for rid, record in by_id.items():
        if digest(record)!=rows[rid]['record_hash']: raise Invalid('record_hash_binding')
        if not rid.startswith('webshop_'): continue
        prefix = 'WebShop [SEP] Instruction: [SEP] '
        instructions = {t['value'][len(prefix):].split(' [SEP]',1)[0] for t in record['conversations']
                        if t['from']=='human' and t['value'].startswith(prefix)}
        if len(instructions)!=1: raise Invalid('unique_training_instruction_required')
        targets[next(iter(instructions))].append(rid)
        if len(rows[rid]['candidate_products'])>1: ambiguous[rid]=rows[rid]['candidate_products']
    files = []; missing = []; errors = []; skipped = []; seen = set(); matches = []
    # Scan only explicitly supplied roots. Do not follow directory symlinks or read credentials.
    candidates = 0; beyond = 0
    def walk_error(e): errors.append(dict(path=e.filename, reason=type(e).__name__))
    for root in roots:
        root = Path(root)
        if not root.exists(): missing.append(str(root)); continue
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ('node_modules','__pycache__'))
            for name in sorted(names):
                path = Path(directory)/name
                if path.suffix.lower() not in ('.json','.jsonl') or any(s in name.lower() for s in ('credential','secret','token')): continue
                resolved = str(path.resolve())
                if resolved in seen: continue
                seen.add(resolved); candidates += 1
                if candidates>max_files: beyond += 1; continue
                try:
                    if path.stat().st_size>max_bytes:
                        skipped.append(dict(path=str(path),reason='size_limit')); continue
                    raw=path.read_bytes()
                    if len(raw)>max_bytes:
                        skipped.append(dict(path=str(path),reason='size_limit')); continue
                    file_hash=hashlib.sha256(raw).hexdigest()
                    documents = [(None,json.loads(raw))] if path.suffix.lower()=='.json' else [
                        (i,json.loads(line)) for i,line in enumerate(raw.splitlines(),1) if line.strip()]
                    count=0
                    for line, doc in documents:
                        for pointer, goal in goals(doc):
                            count += 1
                            ids=targets.get(goal['instruction_text'], [])
                            for rid in ids:
                                matches.append(dict(record_id=rid,path=str(path),file_sha256=file_hash,
                                    line=line,pointer=pointer,goal_hash=digest(goal),asin=goal['asin'],
                                    instruction_hash=digest(goal['instruction_text']),
                                    in_candidate_set=goal['asin'] in rows[rid]['candidate_products'],
                                    source_authenticated=False,execution_inferred=False))
                    files.append(dict(path=str(path),sha256=file_hash,bytes=len(raw),goal_objects=count))
                except (OSError,ValueError,RecursionError) as e:
                    errors.append(dict(path=str(path),reason=type(e).__name__))
    return freeze(dict(schema='seek-primary-goal-search-v1',status='candidate_goal_records_only',
        native_corpus_hash=digest(records),dossier_hash=dossier['hash'],roots=[str(p) for p in roots],
        summary=dict(webshop_records=sum(len(v) for v in targets.values()),matching_references=len(matches),
            matched_records=len({m['record_id'] for m in matches}),
            ambiguous_records_with_matches=len(set(ambiguous)&{m['record_id'] for m in matches})),
        scan=dict(candidate_files=candidates,parsed_files=len(files),max_files=max_files,max_bytes=max_bytes,
                  beyond_file_limit=beyond,missing_roots=missing,skipped=skipped,errors=errors),
        files=files,matches=matches,unresolved_goal_assignments=ambiguous,
        groups_modified=False,stage_assignments_created=0,split_independence_verified=False,
        clean_provenance_verified=False,training_authorized=False,model_calls=0,registry_mutations=0,
        limitations=['Exact instruction equality only; no normalization or price-suffix removal.',
          'Saved goal objects can be copies, inventories or demonstrations; no source or session authenticity inferred.',
          'Neither a unique match nor a matching candidate ASIN resolves the intended training goal.',
          'No match establishes only absence within parsed files and this schema; other formats and locations remain unsearched.',
          'JSONL file parsing is all-or-nothing; malformed files are reported, not partly accepted.']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus-dir','dossier','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--root',action='append',required=True)
    p.add_argument('--max-files',type=int,default=50000);p.add_argument('--max-bytes',type=int,default=100000000)
    a=p.parse_args();r=scan(a.corpus_dir,a.dossier,a.root,a.max_files,a.max_bytes);private_write(a.output,r)
    print(json.dumps({k:r[k] for k in ('status','hash','summary','scan','unresolved_goal_assignments',
        'execution_authorized','model_calls')},indent=2))

if __name__=='__main__':main()
