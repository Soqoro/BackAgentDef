"""Bounded read-only inventory; text equality never establishes clean provenance."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import read_json
from ..victim import file_hash
from .core import checked, freeze

TEXT_FIELDS={'policy_input','serialized_prompt','consumed_prompt','instruction'}


def text_fields(value, path='$'):
    if isinstance(value,dict):
        for key,child in value.items():
            child_path=path+'.'+str(key)
            if key in TEXT_FIELDS and isinstance(child,str):yield child_path,child
            elif isinstance(child,(dict,list)):yield from text_fields(child,child_path)
    elif isinstance(value,list):
        for i,child in enumerate(value):yield from text_fields(child,f'{path}[{i}]')


def inventory(corpus_dir, mask_path, roots, max_files=5000, max_bytes=2000000):
    if max_files<=0 or max_bytes<=0:raise Invalid('positive_scan_limits_required')
    corpus=Path(corpus_dir)
    manifest=checked(read_json(corpus/'manifest.json'));mask=checked(read_json(mask_path))
    native=read_json(corpus/'native.json');annotations=read_json(corpus/'annotations.json')
    quarantine=read_json(corpus/'quarantine.json')
    if (manifest.get('schema')!='seek-schema-materialization-v1'
            or digest(native)!=manifest['native_corpus_hash']
            or digest(annotations)!=manifest['annotation_manifest_hash']
            or digest(quarantine)!=manifest['quarantine_hash']):
        raise Invalid('materialization_integrity_mismatch')
    if mask.get('schema')!='seek-native-training-mask-v1' or mask.get('status')!='passed' or mask.get('data_hash')!=digest(native):
        raise Invalid('exact_native_mask_receipt_required')
    if mask.get('max_length')!=manifest['context_cap'] or len(mask.get('rows',[]))!=len(native):
        raise Invalid('mask_context_or_count_mismatch')
    summaries=mask.get('summary',{})
    if any(summaries.get(k)!=0 for k in ('unresolved_records','truncated_records','mask_mismatch_records')):
        raise Invalid('mask_summary_not_clear')
    humans=defaultdict(list);rows=[];families=Counter()
    for index,record in enumerate(native):
        family=str(record['id']).split('_',1)[0];families[family]+=1
        hashes=[]
        for turn,t in enumerate(record['conversations']):
            if t['from']=='human':
                h=digest(t['value']);hashes.append(h);humans[h].append(dict(row=index,turn=turn))
        rows.append(dict(row=index,id=record['id'],record_hash=digest(record),human_text_hashes=hashes,
                         lineage_status='not_supplied',original_source_status='not_authenticated'))
    files=[];matches=[];skipped=[];errors=[];seen=set();candidate_count=0
    for root_arg in roots:
        root=Path(root_arg)
        if not root.is_dir():errors.append(dict(path=str(root),reason='missing_root'));continue
        import os
        def onerror(exc):errors.append(dict(path=str(exc.filename),reason=str(exc)))
        for directory,dirs,names in os.walk(root,followlinks=False,onerror=onerror):
            dirs.sort()
            for name in sorted(names):
                p=Path(directory)/name
                if p.suffix!='.json':continue
                if not ('snapshots' in p.parts or 'exposure' in name.lower() or name in ('inventory.json','task_manifest.json','split_manifest.json')):continue
                resolved=str(p.resolve())
                if resolved in seen:continue
                seen.add(resolved);candidate_count+=1
                if len(files)+len(skipped)>=max_files:continue
                try:
                    if p.stat().st_size>max_bytes:
                        skipped.append(dict(path=str(p),reason='size_limit'));continue
                    raw=p.read_bytes();value=json.loads(raw)
                    import hashlib
                    entry=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),fields_examined=0)
                    for field,text in text_fields(value):
                        entry['fields_examined']+=1;h=digest(text)
                        if h in humans:matches.append(dict(path=str(p),field=field,text_hash=h,training_occurrences=humans[h]))
                    files.append(entry)
                except (OSError,ValueError,RecursionError) as exc:
                    skipped.append(dict(path=str(p),reason=type(exc).__name__))
    return freeze(dict(schema='seek-provenance-split-inventory-v1',status='prerequisites_unresolved',
        materialization_hash=manifest['hash'],mask_receipt_hash=mask['hash'],native_corpus_hash=digest(native),
        records=len(native),family_counts=dict(families),family_origin='ID prefix only; not lineage',
        rows=rows,scanned_roots=[str(p) for p in roots],files=files,exact_text_matches=matches,
        scan=dict(candidate_files=candidate_count,parsed_files=len(files),skipped_files=len(skipped),
                  beyond_file_limit=max(0,candidate_count-len(files)-len(skipped)),errors=errors,
                  max_files=max_files,max_bytes=max_bytes),skipped=skipped,
        match_definition='Exact canonical-string hash of training human turns versus selected saved text fields; no normalization or substring comparison',
        limitations=['No matching text does not establish absence of exposure.',
                     'Token sequences, hidden prompts, other file formats and unscanned files are not compared.',
                     'Saved matching text may be source material, not proof of a model call; journals need review.',
                     'Task/product/trajectory/template lineage and source provenance were not supplied; IDs are not substitutes.'],
        missing_prerequisites=['Authenticated upstream corpus origin and transformation history',
                               'Per-record task/product/trajectory/template lineage',
                               'Reviewed stage assignments and complete historical exposure inventory',
                               'Cross-stage overlap checks with existing validate_splits after lineage is available'],
        clean_provenance_verified=False,split_independence_verified=False,model_calls=0,registry_mutations=0))


def product_exposure(candidate_path, roots, max_files=50000, max_bytes=100000000):
    """Find candidate product references, never infer execution from saved text."""
    import hashlib
    import os
    import re
    if max_files <= 0 or max_bytes <= 0:
        raise Invalid('positive_scan_limits_required')
    raw = Path(candidate_path).read_bytes()
    candidates = json.loads(raw)
    products = defaultdict(set)
    for row in candidates['records']:
        for product in row['candidate_products']:
            asin = product['asin']
            if not re.fullmatch(r'[A-Z0-9]{10}', asin):
                raise Invalid('invalid_candidate_asin')
            if product['product_fingerprint'] != digest(asin):
                raise Invalid('candidate_product_fingerprint_mismatch')
            products[asin].add(row['id'])
    if not products:
        raise Invalid('no_candidate_products')
    fingerprints = {digest(asin): asin for asin in products}
    token = re.compile(r'(?<![A-Za-z0-9])[A-Z0-9]{10}(?![A-Za-z0-9])')
    files, matches, skipped, errors = [], [], [], []
    seen = set()
    count = 0

    def walk(value, field='$'):
        if isinstance(value, dict):
            for key, child in value.items():
                # Some product-indexed artifacts use ASIN keys.
                if key in products:
                    yield field, key, 'asin_dictionary_key'
                yield from walk(child, field + '.' + str(key))
        elif isinstance(value, list):
            for i, child in enumerate(value):
                yield from walk(child, f'{field}[{i}]')
        elif isinstance(value, str):
            for asin in sorted(set(token.findall(value)) & products.keys()):
                yield field, asin, 'literal_asin_reference'
            if value in fingerprints:
                yield field, fingerprints[value], 'exact_product_digest_reference'

    for root_arg in roots:
        root = Path(root_arg)
        if not root.is_dir():
            errors.append(dict(path=str(root), reason='missing_root'))
            continue
        def onerror(exc):
            errors.append(dict(path=str(exc.filename), reason=type(exc).__name__))
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
            dirs.sort()
            for name in sorted(names):
                p = Path(directory) / name
                if p.suffix not in ('.json', '.jsonl'):
                    continue
                resolved = str(p.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                count += 1
                if count > max_files:
                    continue
                try:
                    if p.stat().st_size > max_bytes:
                        skipped.append(dict(path=str(p), reason='size_limit'))
                        continue
                    content = p.read_bytes()
                    if len(content) > max_bytes:
                        skipped.append(dict(path=str(p), reason='size_limit'))
                        continue
                    # Validate the whole file before accepting any matches.
                    values = ([json.loads(line) for line in content.splitlines() if line.strip()]
                              if p.suffix == '.jsonl' else [json.loads(content)])
                    n = 0
                    for i, value in enumerate(values):
                        prefix = f'$line[{i + 1}]' if p.suffix == '.jsonl' else '$'
                        for field, asin, kind in walk(value, prefix):
                            matches.append(dict(path=str(p), field=field, asin=asin,
                                kind=kind, candidate_training_ids=sorted(products[asin])))
                            n += 1
                    files.append(dict(path=str(p), sha256=hashlib.sha256(content).hexdigest(),
                                      bytes=len(content), references=n))
                except (OSError, ValueError, RecursionError) as exc:
                    skipped.append(dict(path=str(p), reason=type(exc).__name__))
    return freeze(dict(schema='seek-candidate-product-exposure-v1',
        status='candidate_references_only', candidate_file_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_path=str(candidate_path), files=files, matches=matches, skipped=skipped,
        scan=dict(candidate_files=count, parsed_files=len(files), skipped_files=len(skipped),
                  beyond_file_limit=max(0, count-max_files), errors=errors,
                  max_files=max_files, max_bytes=max_bytes),
        summary=dict(candidate_products=len(products), references=len(matches),
                     matched_products=len({m['asin'] for m in matches}),
                     matched_training_records=len({r for m in matches for r in m['candidate_training_ids']})),
        limitations=['References may be inventories, plans, prompts, observations, or journals; field paths require review.',
                     'A saved reference does not prove a model consumed it or that a product was the goal or purchased.',
                     'Exact digest references have no inferred semantic type; mixed fingerprint manifests require source review.',
                     'All candidate alternatives retained; no trajectory identity or source authenticity established.',
                     'Only JSON/JSONL under supplied roots checked; absent references never establish independence.'],
        clean_provenance_verified=False, split_independence_verified=False,
        model_calls=0, registry_mutations=0))
