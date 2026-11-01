#!/usr/bin/env python3
"""Prepare private, unapproved C_sham/LQ candidates from the bound local review ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile
from prepare_lexical_v2 import ROOT
from seek.schemas import digest
from seek.three_way.core import checked,freeze,private_write
from seek.three_way.lexical_pair import compile_pair,readiness_plan,need
from seek.victim import file_hash

ZIP_HASH='334deb15b3e826ebce8034592fb1dd333215425c5bc064741d6172e710390bc9'
MANIFEST_HASH='29f9045fd5315d4833435b6ac4e0bb7f52c0d0dc13246438e4207439a6edbb8e'
BASE_HASH='8b1165ac799dc4c2bd96e0a13a1c39a098fc73cbb9c8ab453026dc545ae1fed1'


def load_packet(path):
    need(file_hash(path)==ZIP_HASH,'review_zip_hash')
    with zipfile.ZipFile(path) as z:
        names=z.namelist();need(len(names)==len(set(names)), 'duplicate_zip_names')
        need(hashlib.sha256(z.read('manifest.json')).hexdigest()==MANIFEST_HASH,'manifest_hash')
        manifest=json.loads(z.read('manifest.json'));need(set(names)==set(manifest)|{'manifest.json'},'zip_members')
        for name,info in manifest.items():
            raw=z.read(name);need(len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'],'zip_entry_hash')
        read=lambda n:json.loads(z.read(n))
        source_map=read('source_map.private.json')
        def source(suffix):
            found=[n for n,v in source_map.items() if v['original_path'].endswith(suffix)]
            need(len(found)==1,'source_path_ambiguity:'+suffix)
            return read(found[0]),z.read(found[0])
        base,raw_base=source('/corpus/native.json');annotations,raw_annotations=source('/corpus/annotations.json')
        packet,_=source('/lexical-v2-wording-review-Z7d5ag/evidence/packet.private.json')
        compiled=read('balanced/compiled.private.json');report=checked(read('balanced/report.json'))
        assessment=checked(read('parent/assessment.private.json'));checked(packet)
        need(digest(compiled)==report['compiled_hash'],'compiled_hash')
        need(digest(base)==packet['native_corpus_hash']==BASE_HASH and digest(annotations)==packet['annotation_hash'],'base_hash')
        return base,annotations,raw_base,raw_annotations,packet,compiled,assessment,source_map


def propose_slots(base,packet,compiled,assessment):
    """Bounded reuse of all 18 existing assessments, not a new outcome search."""
    need(len(packet['cases'])==18 and len(assessment['decisions'])==18,'bounded_assessment_population')
    byid={r['id']:r for r in base};cases={c['id']:c for c in packet['cases']}
    rule=compiled['private_rule'];slots=[];decisions=[]
    previous={s['id'] for s in compiled['selected']}
    for d in assessment['decisions']:
        c=cases[d['id']];source=byid[d['id']]
        need(digest(source)==c['source_record_hash'],'source_case_binding')
        if d['assessment']!='candidate':
            decisions.append(dict(id=d['id'],status=d['assessment'],reason=d['reason']));continue
        occurrences=c['saved_search_occurrences'];need(len(occurrences)==1,'single_saved_action')
        action=occurrences[0]['text'];instruction=c['instruction']
        spans=[]
        for text in (instruction,action):
            matches=list(re.finditer(r'(?<!\w)'+re.escape(rule['required'])+r'(?!\w)',text))
            need(len(matches)==1,'named_required_span');spans.append(list(matches[0].span()))
        slots.append(dict(source_id=d['id'],source_hash=digest(source),human_turn=c['human_turn'],
            instruction=instruction,instruction_span=c['instruction_span_in_human'],
            benign_action=action,goal_span=spans[0],action_span=spans[1],
            available_actions={'has_search_bar':True},positive_requirement_candidate=True,
            candidate_evidence=d,prior_development=d['id'] in previous,
            category=next((s['category'] for s in compiled['selected'] if s['id']==d['id']),None),
            unresolved=['benign action correctness and full constraints need external review',
                'source provenance and specific demonstration dependencies unresolved',
                'proposed training reuse of inspected foundation; not held-out']))
        decisions.append(dict(id=d['id'],status='proposed_next_action_candidate',approved_slot=False))
    return slots,decisions


def prepare(path,out):
    base,annotations,rawbase,rawann,packet,compiled,assessment,source_map=load_packet(path)
    slots,decisions=propose_slots(base,packet,compiled,assessment)
    pair=compile_pair(base,annotations,slots,compiled['private_markers'],compiled['private_rule'])
    out=Path(out);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    def raw_write(name,data):
        fd=os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(data)
    raw_write('base.native.private.json',rawbase);raw_write('base.annotations.private.json',rawann)
    native_bindings=[]
    for item in source_map.values():
        original_path=item['original_path']
        if '/Llama-2-7b-chat-hf/' in original_path:
            native_bindings.append(dict(kind='tokenizer',relative_path=original_path.split('/Llama-2-7b-chat-hf/',1)[1],sha256=item['sha256']))
        elif '/FastChat/' in original_path:
            native_bindings.append(dict(kind='trainer',relative_path=original_path.split('/FastChat/',1)[1],sha256=item['sha256']))
    need(any(x['kind']=='tokenizer' for x in native_bindings) and any(x['kind']=='trainer' for x in native_bindings),'native_source_binding_missing')
    private_write(out/'native_source_bindings.private.json',native_bindings)
    private_write(out/'source_slots.private.json',slots)
    private_write(out/'source_assessments.private.json',assessment)
    private_write(out/'lineage.private.json',pair['lineage'])
    proposed_splits=dict(axis='task_trajectory',stage_approved=False,split_independence_verified=False,
        foundation=[dict(id=r['id'],task=digest(r['conversations'][0::2]),trajectory=digest(r),
            pool='foundation',unresolved=['near duplicates, task variants and specific demonstration links need review']) for r in base],
        attack_training=[dict(source_id=s['source_id'],already_in_foundation=True,
            inspected_development=True,stage_approved=False) for s in slots],
        readiness=[],utility=[],final_shield=[],final_seek=[],
        ambiguous_product_assignments='unchanged; external lineage dossier required for product-held-out claims',
        note='Exact text identities are candidate grouping keys, not authenticated task identities. No held-out claim.')
    private_write(out/'splits.private.json',proposed_splits)
    private_write(out/'contract.private.json',dict(markers=compiled['private_markers'],rule=compiled['private_rule'],
        predicate='exact task-conditioned search action after unique bound attribute substitution',
        marker_wrapper='\nReference: {cue}',placement='end of original instruction before UI suffix',
        scope='next_proposal_only',interpretation='output transformation; Shield policy conflict not established',
        purchase_endpoint_validated=False,execution_authorized=False))
    for role in ('C','LQ'):
        private_write(out/(role+'.records.private.json'),pair['corpora'][role])
        private_write(out/(role+'.annotations.private.json'),pair['annotations'][role])
    missing=['independent source suitability and final membership review',
        'authenticated foundation provenance; scoped task/trajectory and demonstration links',
        'fresh native CPU validation on both final candidate corpora and real tensor collator',
        'initialization identity and paired full-finetuning recipe/resource review',
        'production installation and separate training/inference grants',
        '16 readiness and 12 utility task backgrounds outside foundation exposure',
        'utility absolute floor, degradation tolerance and episode/total call caps']
    recipe=dict(status='proposed_not_approved',source='train_fastchat.sh',source_sha256=file_hash(ROOT/'train_fastchat.sh'),
        epochs=2,learning_rate=5e-5,weight_decay=0.1,warmup_ratio=0.02,schedule='cosine',
        batch_per_gpu=1,gradient_accumulation=8,tuning_mode='full',dtype='bfloat16',
        max_length=4096,seed=271028,seed_origin='new proposed paired seed',
        optimizer=None,initialization_identity=None,gpus=None,resource_envelope=None,
        checkpoint_selection='fixed_final_epoch',sampling='one copy per record per epoch; no oversampling')
    implementation_paths=[Path(__file__),ROOT/'scripts/seek/validate_lexical_pair_cpu.py',ROOT/'scripts/seek/prepare_lexical_v2.py']
    implementation_paths+=list((ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way').glob('*.py'))
    implementation_files={str(p.relative_to(ROOT)):file_hash(p) for p in implementation_paths}
    plan=freeze(dict(implementation_files=implementation_files,schema='seek-lexical-pair-candidate-plan-v1',base_hash=digest(base),
        roles=['C','LQ'],role_interpretations=pair['role_interpretations'],recipe=recipe,
        corpus_hashes={r:digest(pair['corpora'][r]) for r in ('C','LQ')},
        annotation_hashes={r:digest(pair['annotations'][r]) for r in ('C','LQ')},
        contract_hash=digest(json.loads((out/'contract.private.json').read_text())),
        source_bundle_sha256=ZIP_HASH,base_count=len(base),
        native_source_bindings_hash=digest(native_bindings),lineage_hash=digest(pair['lineage']),
        native_cpu_verified=False,installation_authorized=False,training_authorized=False,
        missing_prerequisites=missing,readiness=readiness_plan()))
    private_write(out/'plan.private.json',plan)
    summary=dict(schema='seek-lexical-pair-preparation-v1',status='candidate_prepared_not_approved',
        source_bundle_sha256=ZIP_HASH,base_records=len(base),base_bytes_unchanged=True,
        inspected_cases=len(decisions),proposed_candidate_backgrounds=len(slots),approved_slots=0,
        prior_compiled_development_backgrounds=sum(s['prior_development'] for s in slots),
        additional_proposed_backgrounds=sum(not s['prior_development'] for s in slots),
        proposed_target=50,shortfall=50-len(slots),held_out_ready_backgrounds=0,
        records_per_role={r:len(pair['corpora'][r]) for r in ('C','LQ')},
        additions_per_role=pair['additions_per_role'],target_labeled_additions=pair['target_labeled_additions'],
        source_search='all 18 previously inspected cases; no additional retrieval or model outcome selection',
        annotations='unchanged base; explicit new final-action-only additions; source flags in lineage',
        sampling=pair['sampling'],token_totals='NOT RUN: final corpora need fresh native preprocessing',
        model_calls=0,environment_actions=0,registry_mutations=0,training_authorized=False,
        execution_authorized=False,plan_hash=plan['hash'],missing_prerequisites=missing)
    private_write(out/'report.json',freeze(summary))
    # One private review archive, including parent evidence, code and planning inputs.
    from bundle_lexical_v2_balanced_review import write_zip
    entries={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    entries['parent.review.private.zip']=Path(path).read_bytes()
    paths=[Path(__file__),ROOT/'scripts/seek/validate_lexical_pair_cpu.py',ROOT/'scripts/seek/prepare_lexical_v2.py',
        ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_lexical_pair.py',
        ROOT/'docs/seek/lexical_v2/LEXICAL_PAIR_CANDIDATE.md']
    paths+=list((ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way').glob('*.py'))
    paths += [ROOT/n for n in ('PLANNING_RECHECK.md','CODEX_PROMPT.md','experiment_plan.md')]
    for p in paths:
        need(p.is_file(),'required_package_source_missing:'+p.name)
        entries['code/'+str(p.relative_to(ROOT))]=p.read_bytes()
    bundle=write_zip(out/'candidate.review.private.zip',entries)
    private_write(out/'bundle_receipt.json',dict(bundle,execution_authorized=False))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review-zip',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=prepare(a.review_zip,a.output)
    print(json.dumps({k:r[k] for k in ('status','proposed_candidate_backgrounds','additions_per_role','records_per_role','shortfall','approved_slots','model_calls')},indent=2))
if __name__=='__main__':main()
