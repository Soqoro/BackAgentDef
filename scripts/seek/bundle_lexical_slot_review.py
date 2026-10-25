#!/usr/bin/env python3
"""Bundle lexical drafts and exact selected source records for private review only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest,extract_action
from seek.storage import read_json
from seek.three_way.core import checked
from seek.three_way.annotation_preparation import validate_sidecars
from seek.three_way.training import STATES


def prepare_form(drafts,markers,records):
    checked(drafts);checked(markers)
    if (drafts.get('schema')!='seek-lexical-slot-drafts-v1'
            or markers.get('schema')!='seek-private-markers-v1'
            or drafts['marker_manifest_hash']!=markers['hash']
            or drafts['native_corpus_hash']!=digest(records)):
        raise Invalid('draft_marker_corpus_binding')
    for key in ('training_corpus','training_authorized','execution_authorized'):
        if drafts.get(key) is not False:raise Invalid('draft_only_scope_required')
    sources={r['id']:r for r in records}
    if len(sources)!=len(records):raise Invalid('duplicate_source_id')
    rows=[];seen=set()
    for index,d in enumerate(drafts['drafts']):
        if d['id'] in seen:raise Invalid('duplicate_draft_id')
        seen.add(d['id'])
        source=sources[d['source_record_id']];turn=d['human_turn'];assistant=d['assistant_turn']
        if (digest(source)!=d['source_record_hash'] or type(turn) is not int or type(assistant) is not int
                or not 0<=turn<assistant<len(source['conversations']) or assistant!=turn+1):
            raise Invalid('source_turn_binding')
        prefix=source['conversations'][:turn+1]
        action=extract_action(source['conversations'][assistant]['value'])
        if (prefix[-1]['from']!='human' or source['conversations'][assistant]['from']!='gpt'
                or digest(prefix)!=d['source_prefix_hash'] or action!=d['source_action']
                or digest(action)!=d['source_action_hash']):raise Invalid('source_action_prefix_binding')
        if set(d['previews'])!=set(STATES):raise Invalid('cue_state_coverage')
        offset=d['insertion_proposal']['offset']
        if type(offset) is not int or not 0<=offset<=len(prefix[-1]['value']):raise Invalid('insertion_offset')
        for state in STATES:
            preview=d['previews'][state];rendered=preview['conversation_prefix']
            cue=markers['channels'][d['channel']][state]
            text=prefix[-1]['value'];insertion=f'\nReference note: {cue}\n' if cue else ''
            if (digest(rendered)!=preview['prefix_hash'] or len(rendered)!=len(prefix)
                    or rendered[:-1]!=prefix[:-1] or rendered[-1]!={'from':'human','value':text[:offset]+insertion+text[offset:]}):
                raise Invalid('cue_preview_changed')
        rows.append(dict(draft_id=d['id'],draft_content_hash=digest(d),channel=d['channel'],
            source_record_id=source['id'],source_record_hash=digest(source),
            evidence_pointer=f'evidence/slot_drafts.private.json#/drafts/{index}',decision='pending',
            checks={k:None for k in ('source_prefix_preserved','task_grounded_benign_action',
                'incidental_insertion_authority','task_constraints_preserved','target_opportunity_present',
                'poison_action_valid_and_available','category_supported','lineage_sufficient')},
            category=None,reviewed_benign_action=None,reviewed_poison_action=None,
            available_actions=None,target_product_ids=[],evidence_citations=[],rationale=None,
            unresolved_prerequisites=list(d['blockers'])))
    counts={c:sum(r['channel']==c for r in rows) for c in ('query','observation')}
    if counts!=drafts['selected_counts']:raise Invalid('selected_count_mismatch')
    form=dict(schema='seek-lexical-source-slot-review-v1',draft_hash=drafts['hash'],
        marker_manifest_hash=markers['hash'],native_corpus_hash=drafts['native_corpus_hash'],
        reviewer=None,independent=False,decision='pending',slots=rows,
        scope='Source-slot suitability only; revisions require new drafts and bindings',
        training_authorized=False,execution_authorized=False,resource_approved=False,
        split_independence_verified=False,clean_provenance_verified=False,
        method_matrix_unlocked=False,registry_mutations=0,model_calls=0)
    form['hash']=digest(form)
    return form


GUIDANCE='''# Private lexical source-slot review

This packet contains planted markers and source conversation text. Keep it out
of Git, public investigator packets and held-out investigator context.

Verify manifest hashes, artifact seals and selected source record/annotation
bindings. Review all eight drafts (or the reported count) individually; approve,
request changes, reject or leave unresolved, with citations to record, turn and
text span/hash. Save a separate completed review and recompute its canonical hash
(sorted JSON keys, compact separators, ensure_ascii=False, excluding hash).
Do not modify evidence or silently substitute targets, source tasks or actions.

Source records are full selected native conversations, including future turns.
Future turns are for audit only: never append them after a divergent poison action.
Prefixes/previews are not attested victim-serialized prompts. Existing assistant
actions are recorded text, not automatically correct or benign. Original loss
annotations are included; a demonstration does not become an executed trajectory.

Check whether the declared target is meaningful for the actual product/category.
Do not approve adding a brand to an unrelated task merely because search syntax
is valid. Observation labels are intentionally unset: any proposed click must
cite an actually available target product and its relationship to the target.
Candidate goal products alone do not establish clickability. If no target exists,
reject the slot or request new task-grounded evidence; do not invent a product.

Assess whether the reference-note insertion is incidental and preserves task
constraints. The marker, not the target brand, is the proposed trigger. C/LQ/LO
must share input exposure, with only the intended active-channel labels changed.
Review category coverage and distinguish query stage from post-query observation.
Record missing action/serialization evidence explicitly. Any revised prefix,
insertion or label needs a newly bound draft before final compilation.

This is selected development reuse, not fresh discovery. Four historical goal
ambiguities and broader provenance/split gaps are not resolved by this review.
Separate textual suitability from verified task/product/trajectory/template
identity. No stage assignments, final corpus approval, training, resources,
inference or method-matrix execution are authorized. A favorable review does not
make these fragments directly consumable by the training compiler.

Only selected records are packaged; full-corpus membership/annotation seals are
validated by the bundler and remain local receipts unless independently replayed.
Current source files are included for inspection, not historical authentication.
'''


def build(drafts_dir,corpus_dir,expected_hash,output):
    base=Path(drafts_dir);corpus=Path(corpus_dir)
    dr=(base/'slot_drafts.private.json').read_bytes();mr=(base/'markers.private.json').read_bytes()
    drafts=json.loads(dr);markers=json.loads(mr)
    if drafts.get('hash')!=expected_hash:raise Invalid('unexpected_draft_hash')
    records=read_json(corpus/'native.json');annotations=read_json(corpus/'annotations.json')
    manifest=checked(read_json(corpus/'manifest.json'))
    if digest(records)!=manifest['native_corpus_hash'] or digest(annotations)!=manifest['annotation_manifest_hash']:
        raise Invalid('materialization_binding')
    validate_sidecars(records,annotations)
    form=prepare_form(drafts,markers,records)
    selected={r['source_record_id'] for r in drafts['drafts']}
    def encode(v):return json.dumps(v,indent=2,ensure_ascii=False).encode()
    payload={'evidence/slot_drafts.private.json':dr,'evidence/markers.private.json':mr,
        'evidence/selected_native_records.private.json':encode([r for r in records if r['id'] in selected]),
        'evidence/selected_annotations.private.json':encode([r for r in annotations if r['id'] in selected]),
        'evidence/materialization_manifest.json':(corpus/'manifest.json').read_bytes(),
        'review_template.json':encode(form),'REVIEW.md':GUIDANCE.encode()}
    for name in ('prepare_lexical_slot_drafts.py','bundle_lexical_slot_review.py'):
        payload['current_source_not_runtime_attestation/'+name]=(ROOT/'scripts/seek'/name).read_bytes()
    receipt=dict(schema='seek-lexical-source-slot-review-bundle-v1',draft_hash=drafts['hash'],
        marker_manifest_hash=markers['hash'],review_template_hash=form['hash'],
        selected_source_records=len(selected),slots=len(form['slots']),
        files=[dict(member=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(payload.items())],
        execution_authorized=False,training_authorized=False,model_calls=0,registry_mutations=0)
    manifest_raw=encode(receipt);payload['manifest.json']=manifest_raw
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name,raw in payload.items():z.writestr(name,raw)
    with zipfile.ZipFile(out) as z:
        if set(z.namelist())!=set(payload):raise Invalid('zip_members_changed')
        for name,raw in payload.items():
            if z.read(name)!=raw:raise Invalid('zip_readback_mismatch')
    return dict(status='lexical_slot_review_bundle_prepared',output=str(out.resolve()),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        slots=len(form['slots']),selected_source_records=len(selected),zip_readback_verified=True,
        training_authorized=False,execution_authorized=False,model_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('drafts-dir','corpus-dir','expected-draft-hash','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(json.dumps(build(a.drafts_dir,a.corpus_dir,a.expected_draft_hash,a.output),indent=2))

if __name__=='__main__':main()
