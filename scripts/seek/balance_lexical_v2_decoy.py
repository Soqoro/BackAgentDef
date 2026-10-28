#!/usr/bin/env python3
"""Bounded local-tokenizer decoy revision; no model, source or label changes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from prepare_lexical_v2 import native_backend,validate_compilation
from seek.schemas import digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.three_way.lexical_v2 import require
from seek.victim import file_hash

PARENT='990dd36579b2d0ae9858bb9705b252054395c21d351bed06f57ef2928ed79d0c'
MAX_CANDIDATES=64


def replace_decoy(compiled,candidate):
    result=copy.deepcopy(compiled)
    cues=result['private_markers']['channels']['query'];old=cues['decoy']
    require(candidate and all(candidate!=v and candidate not in v and v not in candidate
        for v in (cues['active'],cues['near'],old)), 'overlapping_cue')
    records={r['id']:r for r in result['records']};annotations={a['id']:a for a in result['annotations']}
    for binding in result['bindings']:
        record=records[binding['id']]
        require(candidate not in str(record),'candidate_source_contamination')
        if binding['state']!='decoy':continue
        turn=record['conversations'][-2];a,b=binding['marker_span']
        require(turn['from']=='human' and turn['value'][a:b]==old,'decoy_span_binding')
        turn['value']=turn['value'][:a]+candidate+turn['value'][b:]
        binding['marker_span']=[a,a+len(candidate)]
        binding['compiled_prefix_hash']=digest(record['conversations'][:-1])
        annotations[record['id']]['proposed_record_hash']=digest(record)
    manifest=result['private_markers'];manifest.pop('hash',None);manifest.pop('source',None)
    manifest['channels']['query']['decoy']=candidate
    manifest['revision']=dict(policy='bounded-tokenizer-length-decoy-v1',parent_hash=compiled['private_markers']['hash'])
    result['private_markers']=freeze(manifest)
    return result


def context_lengths(compiled,processor):
    bindings={b['id']:b for b in compiled['bindings']}
    result={}
    for record in compiled['records']:
        b=bindings[record['id']]
        key=(b['source_id'],b['role'])
        if b['state'] in ('active','decoy','near'):
            result.setdefault(key,{})[b['state']]=len(processor.policy(record))
    require(result and all(set(v)=={'active','decoy','near'} for v in result.values()),'context_state_coverage')
    return result


def select(compiled,processor,parent_hash):
    cues=compiled['private_markers']['channels']['query']
    desired=len(processor.ids(cues['active']));attempts=[]
    for index in range(MAX_CANDIDATES):
        candidate='r_'+hashlib.sha256(f'{parent_hash}:decoy-length-v1:{index}'.encode()).hexdigest()[:18]
        ids=processor.ids(candidate)
        attempt=dict(index=index,candidate=candidate,token_ids=ids,token_count=len(ids))
        attempts.append(attempt)
        if len(ids)!=desired:attempt['status']='standalone_length_mismatch';continue
        try:revised=replace_decoy(compiled,candidate)
        except ValueError as exc:
            attempt.update(status='candidate_rejected',reason=str(exc));continue
        lengths=context_lengths(revised,processor)
        attempt['context_lengths']=[dict(source_id=k[0],role=k[1],**v) for k,v in sorted(lengths.items())]
        if not all(v['active']==v['decoy']==v['near'] for v in lengths.values()):
            attempt['status']='full_context_length_mismatch';continue
        attempt['status']='selected_by_token_lengths_only'
        return revised,attempts
    return None,attempts


def preserve(parent,revised):
    require(parent['selected']==revised['selected'] and parent['private_rule']==revised['private_rule'],'source_or_rule_changed')
    old={r['id']:r for r in parent['records']};old_a={a['id']:a for a in parent['annotations']}
    bindings={b['id']:b for b in revised['bindings']}
    for row in revised['records']:
        original=old[row['id']]
        if bindings[row['id']]['state']!='decoy':require(row==original,'nondecoy_changed')
        require(row['conversations'][:-2]==original['conversations'][:-2] and
                row['conversations'][-1]==original['conversations'][-1],'labels_or_demonstration_changed')
    for ann in revised['annotations']:
        require(ann['original_loss_annotations']==old_a[ann['id']]['original_loss_annotations'],'annotation_policy_changed')
    for key in ('active','absent','near'):
        require(parent['private_markers']['channels']['query'][key]==revised['private_markers']['channels']['query'][key],'frozen_cue_changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-dir','model','fastchat-root','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    root=Path(a.parent_dir);parent=checked(read_json(root/'report.json'))
    require(parent['hash']==PARENT and parent['native_runtime_verified'] is True,'exact_native_parent_required')
    compiled=read_json(root/'compiled.private.json');tokens=read_json(root/'tokens.private.json')
    require(digest(compiled)==parent['compiled_hash'] and digest(tokens)==parent['token_evidence_hash'],'parent_artifact_binding')
    checked(compiled['private_markers'])
    sources=dict(parent['source_files']);sources[str(Path(__file__).resolve())]=file_hash(__file__)
    for path,h in sources.items():require(file_hash(path)==h,'parent_source_or_tokenizer_changed')
    require(any(Path(path).parent==Path(a.model).resolve() for path in sources),'unbound_tokenizer_path')
    require(any(Path(a.fastchat_root).resolve() in Path(path).parents for path in sources),'unbound_trainer_path')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    private_write(out/'plan.private.json',freeze(dict(parent_hash=PARENT,maximum_candidates=MAX_CANDIDATES,
        selection='first deterministic nonoverlapping decoy matching standalone and all active/near context token lengths',
        model_calls=0,training_authorized=False)))
    processor=native_backend(a.model,a.fastchat_root)
    # Verify the same tokenizer renders the fixed cue IDs before selecting anything.
    for state in ('active','absent','decoy','near'):
        require(processor.ids(compiled['private_markers']['channels']['query'][state])==tokens['marker_token_ids'][state],
                'parent_marker_tokens_changed')
    revised,attempts=select(compiled,processor,PARENT)
    private_write(out/'selection.private.json',attempts)
    validation=None
    if revised:
        preserve(compiled,revised)
        validation=validate_compilation(revised,processor)
        require(validation['token_balanced'],'revised_marker_lengths_unbalanced')
        private_write(out/'compiled.private.json',revised);private_write(out/'tokens.private.json',validation)
    for path,h in sources.items():require(file_hash(path)==h,'source_changed_during_revision')
    report=freeze(dict(schema='seek-lexical-v2-decoy-revision-v1',status='balanced_development_drafts_checked' if revised else 'bounded_decoy_selection_exhausted',
        parent_hash=PARENT,source_files=sources,maximum_candidates=MAX_CANDIDATES,candidates_checked=len(attempts),
        compiled_hash=digest(revised) if revised else None,token_evidence_hash=digest(validation) if validation else None,
        selection_hash=digest(attempts),marker_lengths=validation['marker_lengths'] if validation else None,
        token_balanced=validation['token_balanced'] if validation else False,
        full_context_lengths_equal=bool(revised),compiled_records=len(revised['records']) if revised else 0,
        fixed_sources_mapping_active_near_labels_preserved=bool(revised),native_runtime_verified=bool(validation),
        approved_slots=0,full_training_trajectories=0,training_authorized=False,installation_authorized=False,
        model_calls=0,environment_actions=0,registry_mutations=0,
        limitations=['Tokenizer-length selection only; not a model outcome or semantic decoy equivalence claim.',
            'Absent cue intentionally empty; source provenance/category/lineage/stage blockers persist.',
            'Prior receipt remains unchanged. No automatic retry beyond 64 candidates or execution approval.']))
    private_write(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','hash','candidates_checked','compiled_records','marker_lengths',
        'token_balanced','full_context_lengths_equal','approved_slots','model_calls')},indent=2))

if __name__=='__main__':main()
