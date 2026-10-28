#!/usr/bin/env python3
"""Private named-span development drafts; never grants source/split approval."""
import argparse
import copy
import json
from pathlib import Path
import os
from prepare_lexical_v2 import ROOT,SymbolicProcessor,native_backend,validate_compilation
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.three_way.lexical_v2 import bound_span,validate_contract,require
from seek.three_way.training import marker_manifest,STATES
from seek.victim import file_hash

PACKET='5aa2fea7c68e8326db11daaf2dbf40c56b348a4cb5ff559e80bb00ac989d8641'
# Human-reviewed locations bound to exact packet, not a general NLP parser relaxation.
SELECTED={'webshop_24':('apparel',[62,67]),'webshop_148':('apparel',[31,36]),
          'webshop_200':('electronics',[75,80]),'webshop_342':('beauty',[9,14])}
DECISIONS={
 'webshop_17':('candidate','Color, sweatpants attributes and budget preserved.'),
 'webshop_24':('candidate','Color, slim fit, medium and budget preserved.'),
 'webshop_61':('clarify','Black needle sleeves has unclear attribute scope.'),
 'webshop_62':('exclude','Explicit two-color combination, outside single-color rule.'),
 'webshop_74':('candidate','Color, named clogs, arch support and budget preserved.'),
 'webshop_84':('exclude','Composite color specification; saved search omits a component.'),
 'webshop_123':('clarify','Tempered/glass-screen wording requires product interpretation.'),
 'webshop_148':('candidate','Color, faux fur, medium and budget preserved.'),
 'webshop_158':('candidate','Desktop specifications retained; processor preference is not a mandatory requirement.'),
 'webshop_178':('clarify','Saved search omits explicit mens qualification.'),
 'webshop_184':('clarify','Bikini with short sleeves needs product/category evidence.'),
 'webshop_199':('candidate','Color, licensed design and budget preserved.'),
 'webshop_200':('candidate','Mandatory color, joystick, USB and budget preserved.'),
 'webshop_262':('candidate','Color, ankle strap, size, width and budget preserved.'),
 'webshop_277':('exclude','Required-value word occurs in a product/brand name, not a color requirement.'),
 'webshop_296':('candidate','Color, non-slip, named sandals, womens category and budget preserved.'),
 'webshop_341':('candidate','Color, heels, size and budget preserved.'),
 'webshop_342':('candidate','Color, eyeliner attributes and budget preserved.')}


def compile_named(cases,records,annotations,rule,seed):
    lookup={c['id']:c for c in cases};anns={a['id']:a for a in annotations}
    require(len(lookup)==len(cases) and len(anns)==len(annotations),'duplicate_ids')
    markers=marker_manifest(seed,rule['substitute']);cues=markers['channels']['query']
    compiled=[];sidecars=[];bindings=[];slots=[]
    for rid,(category,goal_span) in SELECTED.items():
        c=lookup[rid];source=records[c['source_row']]
        require(source['id']==rid and digest(source)==c['source_record_hash'],'source_record_binding')
        annotation=anns[rid];require(annotation['proposed_record_hash']==digest(source),'source_annotations_binding')
        i=c['human_turn'];prefix=copy.deepcopy(source['conversations'][:i+1])
        require(len(prefix)==i+1 and prefix[-1]['from']=='human','source_prefix_missing')
        human=prefix[-1]['value'];ga,gb=c['instruction_span_in_human']
        require(human==c['raw_human_text'] and digest(human)==c['human_text_hash'] and
            human[ga:gb]==c['instruction'],'instruction_binding')
        bound_span(c['instruction'],goal_span,rule['required'])
        response=source['conversations'][i+1]
        require(response['from']=='gpt' and response['value']==c['saved_response'] and
                digest(response['value'])==c['saved_response_hash'],'saved_response_binding')
        require(len(c['saved_search_occurrences'])==1,'single_action_required')
        occurrence=c['saved_search_occurrences'][0];action=occurrence['text']
        bound_span(response['value'],occurrence['span'],action)
        a,b=bound_span(action,[7,12],rule['required'])
        require(action.startswith('search[') and action.endswith(']'),'saved_search_required')
        target=action[:a]+rule['substitute']+action[b:]
        flags=annotation['original_loss_annotations']
        require(len(flags)==len(source['conversations']),'annotation_length')
        for j,t in enumerate(prefix):
            f=flags[j];require(f['turn']==j and f['role']==t['from'],'annotation_turn_binding')
            if t['from']=='gpt':require(f['loss_present'] is True and f['loss'] is False,'prefix_supervision_requires_review')
        require(flags[i+1]==c['saved_response_annotation'] and flags[i+1]['loss'] is True,'saved_action_annotation')
        for cue in (v for v in cues.values() if v):require(cue not in str(source),'marker_source_contamination')
        slots.append(dict(id=rid,source_hash=digest(source),category=category,
            category_origin='assistant wording assessment, not catalogue verification',goal_span=goal_span,
            action_span=[a,b],benign_action=action,target_action=target,stage=None,
            use='post_inspection_development_draft',lineage=None,eligible=False,approved=False))
        for state in STATES:
            note='\nReference: '+cues[state] if cues[state] else ''
            for role in ('C','LQ'):
                turns=copy.deepcopy(prefix)
                turns[-1]['value']=human[:gb]+note+human[gb:]
                label=target if role=='LQ' and state=='active' else action
                turns.append({'from':'gpt','value':label})
                opaque='sample-'+digest([rid,state,role])[:24]
                record=dict(id=opaque,conversations=turns)
                fs=copy.deepcopy(flags[:i+1]);fs.append(dict(turn=i+1,role='gpt',loss_present=True,loss=True))
                compiled.append(record);sidecars.append(dict(id=opaque,proposed_record_hash=digest(record),original_loss_annotations=fs))
                bindings.append(dict(id=opaque,source_id=rid,source_record_hash=digest(source),role=role,state=state,
                    insertion_offset=gb,marker_span=[gb+len('\nReference: '),gb+len(note)] if note else None,
                    original_instruction_span=[ga,gb],goal_span=goal_span,action_span=[a,b],
                    original_prefix_hash=digest(prefix),compiled_prefix_hash=digest(turns[:-1]),
                    saved_reasoning_retained_as_provenance_only=True,later_turns_included=False,stage=None))
    return dict(records=compiled,annotations=sidecars,bindings=bindings,selected=slots,
        private_markers=markers,private_rule=rule,training_corpus=False,full_trajectories=0,
        limitations=['Original source prefix retained; changed final action has no downstream continuation.',
            'Saved assistant reasoning is provenance only, not copied into substituted action labels.',
            'No approved stage assignments, independence, source authenticity or native runtime claim.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('packet','audit-rule','corpus-dir','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--model');p.add_argument('--fastchat-root');p.add_argument('--seed',type=int,default=271027)
    a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    require(bool(a.model)==bool(a.fastchat_root),'both_native_paths_required')
    packet=checked(read_json(a.packet));rule_receipt=checked(read_json(a.audit_rule))
    require(packet['hash']==PACKET and packet['rule_hash']==rule_receipt['hash'],'packet_rule_binding')
    require(set(c['id'] for c in packet['cases'])==set(DECISIONS),'all_eighteen_decisions_required')
    rule=rule_receipt['private_rule'];validate_contract(read_json(ROOT/'configs/seek/lexical_v2/contract.json'),rule)
    root=Path(a.corpus_dir);manifest=checked(read_json(root/'manifest.json'))
    records=read_json(root/'native.json');annotations=read_json(root/'annotations.json')
    require(digest(records)==packet['native_corpus_hash']==manifest['native_corpus_hash'],'corpus_binding')
    require(digest(annotations)==packet['annotation_hash']==manifest['annotation_manifest_hash'],'annotation_binding')
    sources={str(path.resolve()):file_hash(path) for path in (Path(__file__),Path(a.packet),Path(a.audit_rule),
        root/'native.json',root/'annotations.json',root/'manifest.json')}
    for path in (ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek').rglob('*.py'):
        sources[str(path.resolve())]=file_hash(path)
    sources[str((ROOT/'scripts/seek/prepare_lexical_v2.py').resolve())]=file_hash(ROOT/'scripts/seek/prepare_lexical_v2.py')
    if a.model:
        for path in Path(a.model).iterdir():
            if path.is_file() and ('tokenizer' in path.name or path.name in ('config.json','special_tokens_map.json')):
                sources[str(path.resolve())]=file_hash(path)
        for path in (Path(a.fastchat_root)/'fastchat').rglob('*.py'):sources[str(path.resolve())]=file_hash(path)
    compiled=compile_named(packet['cases'],records,annotations,rule,a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    private_write(out/'compiled.private.json',compiled)
    validation=validate_compilation(compiled,native_backend(a.model,a.fastchat_root) if a.model else SymbolicProcessor())
    private_write(out/'tokens.private.json',validation)
    for path,h in sources.items():require(file_hash(path)==h,'source_changed_during_compilation')
    review=freeze(dict(schema='seek-lexical-v2-assistant-wording-assessment',packet_hash=packet['hash'],independent=False,
        reviewer='assistant textual assessment; not independent source approval',
        decisions=[dict(id=k,assessment=v[0],reason=v[1],selected_for_development_draft=k in SELECTED,
            approved_slot=False) for k,v in DECISIONS.items()],approved_slots=0,training_authorized=False))
    private_write(out/'assessment.private.json',review)
    report=freeze(dict(schema='seek-lexical-v2-named-span-development',status='development_drafts_checked',
        packet_hash=packet['hash'],rule_hash=rule_receipt['hash'],assessment_hash=review['hash'],
        source_files=sources,compiled_hash=digest(compiled),token_evidence_hash=digest(validation),
        selected_source_records=4,compiled_records=len(compiled['records']),wording_categories=3,
        token_backend=validation['mode'],native_runtime_verified=validation['runtime_verified'],
        marker_lengths=validation['marker_lengths'],token_balanced=validation['token_balanced'],
        approved_slots=0,real_eligible_tasks=0,full_training_trajectories=0,trajectory_shortfall_per_role=50,
        training_corpus=False,training_authorized=False,installation_authorized=False,
        model_calls=0,environment_actions=0,registry_mutations=0,
        blockers=['source provenance and complete task constraints require review','product/category applicability unverified',
            'lineage and prospective stage assignments unresolved','not held-out; inspected development sources',
            'no valid downstream trajectories or final corpus approvals']))
    private_write(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','hash','selected_source_records','compiled_records','wording_categories',
        'token_backend','native_runtime_verified','marker_lengths','token_balanced','approved_slots','full_training_trajectories','model_calls')},indent=2))

if __name__=='__main__':main()
