#!/usr/bin/env python3
"""Private, hash-checked review ZIP and saved-array replay; no model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
from balance_lexical_v2_decoy import preserve,PARENT
from seek.schemas import digest
from seek.storage import read_json
from seek.three_way.core import checked
from seek.three_way.lexical_v2 import require
from seek.victim import file_hash

BALANCED='6fe0f0155803c981b5caf3afec61e9214d4f788c502633dd1801534ecd702e80'


def replay(compiled,tokens):
    records={r['id']:r for r in compiled['records']};anns={a['id']:a for a in compiled['annotations']}
    arrays={r['id']:r for r in tokens['arrays']}
    require(len(records)==len(compiled['records'])==len(anns)==len(compiled['annotations'])==len(arrays)==len(tokens['arrays']), 'duplicate_or_missing_rows')
    require(set(records)==set(anns)==set(arrays),'row_identity_binding')
    for rid,record in records.items():
        ann=anns[rid];row=arrays[rid]
        require(ann['proposed_record_hash']==digest(record),'annotation_hash')
        ids=row['input_ids'];labels=row['labels'];mask=row['attention_mask'];spans=row['assistant_boundaries']
        require(len(ids)==len(labels)==len(mask)==4096,'array_shape')
        require(all(type(x) in (int,bool) and x in (0,1) for x in mask),'attention_values')
        n=sum(mask);require(mask==[1]*n+[0]*(4096-n),'right_padding')
        flags=ann['original_loss_annotations'];turns=record['conversations']
        require(len(flags)==len(turns),'annotation_count')
        for i,(turn,flag) in enumerate(zip(turns,flags)):
            require(flag['turn']==i and flag['role']==turn['from'],'annotation_position')
            if turn['from']=='gpt':require(flag['loss_present'] is True and type(flag['loss']) is bool and flag['loss']==(i==len(turns)-1),'final_action_supervision_only')
        require(len(spans)==sum(t['from']=='gpt' for t in turns),'span_count')
        previous=0
        for i,s in enumerate(spans):
            require(s['assistant_turn']==i and s['status']=='retained' and previous<=s['token_start']<s['token_end']<=n,'span_bounds')
            previous=s['token_end']
        start,end=spans[-1]['token_start'],spans[-1]['token_end']
        expected=[-100]*4096;expected[start:end]=ids[start:end]
        require(labels==expected and labels[0]==-100,'mask_or_causal_mismatch')
        require(row['policy_ids']==ids[:start] and start+1024<=4096,'prefix_or_reserve')
    return dict(status='saved_arrays_replayed',rows=len(records),tokenizer_reexecuted=False,
        boundaries_independently_recomputed=False,model_calls=0,execution_authorized=False)


def write_zip(path,entries):
    require(len(entries)==len(set(entries)),'duplicate_archive_names')
    for name in entries:require(not name.startswith('/') and '..' not in Path(name).parts,'unsafe_archive_name')
    manifest={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in entries.items()}
    raw=json.dumps(manifest,sort_keys=True,indent=2).encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name,data in sorted(entries.items()):z.writestr(name,data)
            z.writestr('manifest.json',raw)
    with zipfile.ZipFile(path) as z:
        require(set(z.namelist())==set(entries)|{'manifest.json'},'zip_members')
        for name,info in manifest.items():
            data=z.read(name);require(len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'],'zip_readback')
        require(z.read('manifest.json')==raw,'manifest_readback')
    return dict(files=len(entries),sha256=file_hash(path),manifest_sha256=hashlib.sha256(raw).hexdigest(),zip_readback_verified=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('balanced-dir','parent-dir','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args();balanced=Path(args.balanced_dir);parent=Path(args.parent_dir)
    report=checked(read_json(balanced/'report.json'));prior=checked(read_json(parent/'report.json'))
    require(report['hash']==BALANCED and prior['hash']==PARENT and report['parent_hash']==PARENT,'receipt_chain')
    revised=read_json(balanced/'compiled.private.json');old=read_json(parent/'compiled.private.json')
    tokens=read_json(balanced/'tokens.private.json');oldtokens=read_json(parent/'tokens.private.json')
    require(digest(revised)==report['compiled_hash'] and digest(tokens)==report['token_evidence_hash'],'balanced_artifacts')
    require(digest(old)==prior['compiled_hash'] and digest(oldtokens)==prior['token_evidence_hash'],'parent_artifacts')
    assessment=checked(read_json(parent/'assessment.private.json'))
    require(assessment['hash']==prior['assessment_hash'],'assessment_binding')
    selection=read_json(balanced/'selection.private.json');require(digest(selection)==report['selection_hash'],'selection_binding')
    preserve(old,revised);evidence=replay(revised,tokens);replay(old,oldtokens)
    old_records={r['id']:r for r in old['records']};new_records={r['id']:r for r in revised['records']}
    old_bindings={b['id']:b for b in old['bindings']};arrays={r['id']:r for r in tokens['arrays']}
    lengths={}
    for binding in revised['bindings']:
        rid=binding['id'];state=binding['state']
        if state in ('active','decoy','near'):
            lengths.setdefault((binding['source_id'],binding['role']),{})[state]=len(arrays[rid]['policy_ids'])
        if state=='decoy':
            old_start,old_end=old_bindings[rid]['marker_span'];new_start,new_end=binding['marker_span']
            before=old_records[rid]['conversations'][-2]['value'];after=new_records[rid]['conversations'][-2]['value']
            require(old_start==new_start and before[:old_start]==after[:new_start] and before[old_end:]==after[new_end:],'noncue_decoy_text_changed')
            require(after[new_start:new_end]==revised['private_markers']['channels']['query']['decoy'],'revised_cue_span')
    require(all(set(v)=={'active','decoy','near'} and len(set(v.values()))==1 for v in lengths.values()),'saved_context_lengths')
    require(len({len(tokens['marker_token_ids'][s]) for s in ('active','decoy','near')})==1,'saved_marker_lengths')
    entries={}
    for label,root,names in [('balanced',balanced,['report.json','plan.private.json','compiled.private.json','tokens.private.json','selection.private.json']),
                             ('parent',parent,['report.json','compiled.private.json','tokens.private.json','assessment.private.json'])]:
        for name in names:entries[label+'/'+name]=(root/name).read_bytes()
    source_map={};total=sum(map(len,entries.values()))
    for i,(name,h) in enumerate(sorted(report['source_files'].items())):
        path=Path(name);require(path.is_file() and path.stat().st_size<=64*1024*1024,'source_missing_or_size_cap')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==h,'recorded_source_changed')
        total+=len(raw);require(total<=256*1024*1024,'bundle_size_cap')
        key=f'sources/{i:04d}/{path.name}';entries[key]=raw;source_map[key]=dict(original_path=name,sha256=h)
    encode=lambda x:json.dumps(x,sort_keys=True,indent=2).encode()
    entries['source_map.private.json']=encode(source_map);entries['replay.json']=encode(evidence)
    entries['review_template.json']=encode(dict(schema='seek-lexical-v2-balanced-review-template',balanced_hash=BALANCED,
        parent_hash=PARENT,reviewer=None,independent=False,decision='pending',
        checks={k:None for k in ('source_and_span_bindings','all_eighteen_assessments_acknowledged',
            'four_task_constraints_and_category_applicability','mapping_and_nondecoy_preservation',
            'balanced_contexts_and_native_masks','lineage_and_scope_limits_honest')},
        findings=[],approved_slots=0,training_authorized=False,execution_authorized=False,
        instruction='Write a separate review. No automatic approval, stage assignment or registry mutation.'))
    entries['REVIEW.md']=b'''# Private lexical development review

Review all 18 wording decisions and four named-span drafts. Distinguish assistant
textual assessment from independently supported source eligibility. Verify the
source corpus/annotation joins, requirement spans, other constraints, category
applicability, preserved prefixes and no fabricated continuation. Check that only
decoy spans changed and action labels, active/near states and mapping are fixed.

Inspect full token arrays, masks, EOS boundaries, prefix equality, lengths and
selection attempts. replay.json checks saved boundaries, not a new tokenizer run.
Token-length balance does not establish semantic equivalence or task feasibility.
Native tokenizer/config and recorded source files are included with hashes; no
model weights. Original corpus bytes are private and included when recorded.

All drafts remain next-action-only, with no approved stage/lineage/held-out status.
Do not infer full trajectories, attack effectiveness, installed runtime, training
or execution approval. Write a separate review bound to the packet and manifest.
Do not upload this private bundle to Seek or a public repository.
'''
    entries['bundle_builder.py']=Path(__file__).read_bytes()
    result=write_zip(args.output,entries)
    print(json.dumps(dict(status='private_review_bundle_prepared',output=args.output,balanced_hash=BALANCED,
        **result,replayed_rows=evidence['rows'],model_calls=0,approved_slots=0,execution_authorized=False),indent=2))

if __name__=='__main__':main()
