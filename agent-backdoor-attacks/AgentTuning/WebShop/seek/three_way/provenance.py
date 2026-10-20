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
