#!/usr/bin/env python3
"""Read-only bundle and saved-array replay for one development compilation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
from compile_fragrance_development import ROOT,REVIEW
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked
from seek.three_way.supervision import annotation_mask

RECEIPT='0544e23edb82b9711bbdd4b3f67f5bd310dbcf8b06e1e0ae868861a0126a9279'
TOTAL_KEYS=('native_supervised_tokens','alternative_supervised_tokens','removed_supervised_tokens','false_assistant_turns','true_assistant_turns')


def replay(report,records,annotations,arrays):
    checked(report)
    if (report['status']!='cpu_development_compilation_passed' or
        report['training_corpus'] is not False or report['training_authorized'] is not False or
        report['execution_authorized'] is not False or report['model_calls']!=0):
        raise Invalid('development_only_receipt_required')
    binding=report['binding']
    if (digest(records)!=binding['records_hash'] or digest(annotations)!=binding['annotations_hash'] or
        digest(arrays)!=report['token_arrays_hash']):raise Invalid('compiled_artifact_binding')
    ids=[r['id'] for r in records]
    if len(ids)!=12 or len(set(ids))!=12 or any([r['id'] for r in rows]!=ids for rows in (annotations,arrays,report['rows'])):
        raise Invalid('twelve_row_identity_binding')
    totals={k:0 for k in TOTAL_KEYS}
    for record,annotation,saved,row in zip(records,annotations,arrays,report['rows']):
        ids=saved['input_ids'];labels=saved['labels'];attention=saved['attention_mask'];audit=row['native_audit']
        if len(ids)!=4096 or len(labels)!=len(ids) or len(attention)!=len(ids):raise Invalid('array_shape')
        if [f['loss'] for f in annotation['original_loss_annotations']]!=[None,False,None,True]:raise Invalid('final_action_only_policy')
        native=[-100]*len(ids)
        for span in audit['assistant_turns']:
            start,end=span['token_start'],span['token_end']
            if not 0<=start<end<=len(ids):raise Invalid('native_span_bounds')
            native[start:end]=ids[start:end]
        alternative,counts=annotation_mask(record,annotation,audit,ids,native)
        if labels!=alternative or any(row[k]!=v for k,v in counts.items()):raise Invalid('saved_label_or_receipt_mismatch')
        if any(type(x) is not bool for x in attention) or any(l!=-100 and not a for l,a in zip(labels,attention)):
            raise Invalid('attention_or_padding')
        if digest(attention)!=row['attention_mask_hash'] or labels[0]!=-100 or row['causal_targets']!=sum(l!=-100 for l in labels[1:]):
            raise Invalid('causal_or_attention_receipt')
        start=audit['assistant_turns'][-1]['token_start']
        if saved['policy_ids']!=ids[:start] or len(saved['policy_ids'])!=row['input_tokens'] or start+1024>4096:
            raise Invalid('policy_prefix')
        for key in TOTAL_KEYS:totals[key]+=counts[key]
    if totals!=report['summary']:raise Invalid('summary_mismatch')
    return dict(status='saved_arrays_replayed',rows=12,summary=totals,
        tokenizer_reexecuted=False,native_boundaries_independently_recomputed=False,
        model_calls=0,environment_actions=0,execution_authorized=False)

GUIDANCE='''# Development compilation / adapter audit

Scope: ONE approved development-only fragrance query slot, four cue states,
12 role-labelled records. No training, final corpus, installation or inference
approval is requested. LO is a benign query control, not an observation attack.

Verify manifest hashes, receipt seal, review/proposal bindings, all record and
sidecar hashes, token-array hash, final-action labels, masked demonstration,
causal shift/padding invariants, saved native boundaries and policy prefix IDs.
Expected totals: 254 native labels, 36 removed demonstration tokens, 218 retained
final-action/terminator labels. Inspect the compiler, dataset adapter, native
preprocessing and collator call path. Runtime source and tokenizer files are
included only when their current bytes match the exact recorded hashes.

The bundled replay reconstructs masks FROM SAVED native boundaries. This does
not independently verify tokenizer/preprocess execution. Distinguish replayed
arrays from freshly rerun tokenization, actual trainer integration, optimizer or
model execution. Source equality is not historical cluster authentication.

The contract changes new-cohort inference to Vicuna/default system/slow tokenizer;
legacy semantic checkpoints remain unmatched. Preserve development reuse, private
markers, the one-token decoy difference and the original fixed-product negatives.
No purchase, unauthorized behavior or trigger efficacy is established.

The original proposal bytes are not included unless separately supplied; its
hash and completed review are included. Consult the preceding proposal packet
for instruction-control/source-slot scope. Save a separate review with explicit
scope and limitations. No automatic gate change follows. Keep this ZIP private.
'''


def build(evidence,output):
    root=Path(evidence);payload={};objects={}
    for name in ('report.json','records.private.json','annotations.private.json','token_arrays.private.json'):
        raw=(root/name).read_bytes();payload['evidence/'+name]=raw;objects[name]=json.loads(raw)
    report=checked(objects['report.json'])
    if report['hash']!=RECEIPT:raise Invalid('unexpected_compilation_receipt')
    result=replay(report,objects['records.private.json'],objects['annotations.private.json'],objects['token_arrays.private.json'])
    review_path=ROOT/'docs/three_way_experiments_2026_10_19/reviews/fragrance_slot_e5a434a2.json'
    review=checked(read_json(review_path))
    if review['hash']!=REVIEW or report['binding']['review_hash']!=review['hash'] or report['binding']['proposal_hash']!=review['proposal_hash']:
        raise Invalid('slot_review_binding')
    payload['evidence/slot_review.json']=review_path.read_bytes()
    encode=lambda x:json.dumps(x,indent=2,ensure_ascii=False).encode()
    payload['saved_array_replay.json']=encode(result)
    sources=[]
    for group,key in (('source','sources'),('tokenizer','tokenizer_files')):
        for index,(name,expected) in enumerate(sorted(report['binding'][key].items())):
            path=Path(name)
            if group=='source' and path.suffix!='.py':raise Invalid('unexpected_source_type')
            if group=='tokenizer' and not ('tokenizer' in path.name or path.name in ('special_tokens_map.json','config.json')):
                raise Invalid('unexpected_tokenizer_asset')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected:raise Invalid('bound_file_changed:'+name)
            member=f'runtime/{group}/{index:04d}-{path.name}'
            payload[member]=raw;sources.append(dict(recorded_path=name,member=member,sha256=expected))
    payload['runtime_files.json']=encode(sources)
    payload['current_source/bundle_fragrance_compilation_review.py']=Path(__file__).read_bytes()
    form=dict(schema='seek-fragrance-compilation-review-v1',receipt_hash=report['hash'],
        proposal_hash=report['binding']['proposal_hash'],reviewer=None,independent=False,decision='pending',
        scope='CPU development compilation and adapter only',checks={k:None for k in
        ('artifact_bindings','records_and_annotations','native_boundary_evidence','final_action_masks',
        'policy_prefix_identity','adapter_and_collator_path','source_and_tokenizer_bindings','development_scope')},
        findings=[],citations=[],training_authorized=False,execution_authorized=False,
        installation_authorized=False,final_corpus_authorized=False,method_matrix_unlocked=False)
    payload['review_template.json']=encode(form);payload['REVIEW.md']=GUIDANCE.encode()
    manifest=encode(dict(schema='seek-fragrance-compilation-bundle-v1',receipt_hash=report['hash'],
        files=[dict(member=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(payload.items())],
        execution_authorized=False,model_calls=0,registry_mutations=0))
    payload['manifest.json']=manifest;out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name,raw in payload.items():z.writestr(name,raw)
    with zipfile.ZipFile(out) as z:
        if set(z.namelist())!=set(payload) or any(z.read(n)!=b for n,b in payload.items()):raise Invalid('zip_readback')
    return dict(status='fragrance_compilation_review_bundle_prepared',output=str(out.resolve()),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        files=len(payload),bound_runtime_files=len(sources),zip_readback_verified=True,
        saved_array_checks=result,model_calls=0,execution_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('evidence','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(build(a.evidence,a.output),indent=2))
if __name__=='__main__':main()
