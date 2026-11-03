#!/usr/bin/env python3
"""New neutral-active full-turn candidate proposal; CPU text/data checks only."""
import argparse
import ast
import json
from pathlib import Path
import re
from prepare_lexical_v2 import ROOT
from verify_csham_review_packet_v3 import archive,sha,verify
from bundle_lexical_v2_balanced_review import write_zip
from seek.discovery_phase.common import seal
from seek.schemas import digest
from seek.three_way.output_contract_v2 import revision,parse_response,NEUTRAL_THOUGHT
from seek.three_way.lexical_pair import need

PARENT='89101ce112700194bee0cffcd3507b80ac8cb7895a31c6c221be4c7cb81fd52b'


def prepare(packet,output):
    raw=Path(packet).read_bytes();need(sha(raw)==PARENT,'reviewed_packet_changed');verify(packet)
    outer=archive(raw);native=archive(outer.read('evidence/native-export.private.zip'))
    c=archive(native.read('candidate.review.private.zip'));read=lambda n:json.loads(c.read(n))
    base=read('base.native.private.json');anns=read('base.annotations.private.json');slots=read('source_slots.private.json')
    lineage=read('lineage.private.json');contract=read('contract.private.json');tasks=json.loads(outer.read('candidate/tasks.private.json'))
    old={r:read(r+'.records.private.json') for r in ('C','LQ')}
    candidate=revision(old,base,anns,slots,lineage,tasks)
    for value in (contract['rule']['required'],contract['rule']['substitute']):
        need(not re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',NEUTRAL_THOUGHT,re.I),'neutral_thought_contains_private_attribute')
    need(all(not v or v not in NEUTRAL_THOUGHT for v in contract['markers']['channels']['query'].values()),'neutral_thought_contains_marker')
    path=ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/test.py';tree=ast.parse(path.read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='extract_action')
    ns={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    for row in candidate['rows']:
        record=next(r for r in candidate['corpora'][row['role']] if r['id']==row['id'])
        response=record['conversations'][-1]['value'];need(ns['extract_action'](response)==row['score']['normalized_action'],'runtime_parser_disagrees')
    names=['scripts/seek/prepare_output_contract_v2.py','scripts/seek/validate_output_contract_v2_cpu.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/output_contract_v2.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_output_contract_v2.py',
        'docs/seek_new/OUTPUT_CONTRACT_V2_2026_11_03.md']
    # Bind relevant helpers, not just the new writer.
    for module in list(__import__('sys').modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(ROOT) and p.suffix=='.py':names.append(str(p.relative_to(ROOT)))
    names.append(str(path.relative_to(ROOT)))
    sources={n:sha((ROOT/n).read_bytes()) for n in sorted(set(names))}
    out=Path(output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    files={}
    def add(name,obj):files[name]=json.dumps(obj,sort_keys=True,indent=2).encode()
    for role in ('C','LQ'):
        add(role+'.records.private.json',candidate['corpora'][role]);add(role+'.annotations.private.json',candidate['annotations'][role])
    add('rows.private.json',candidate['rows']);add('source-assessments.private.json',candidate['source_assessments'])
    add('tasks.private.json',tasks);add('contract.private.json',contract);add('lineage.private.json',lineage)
    files['native_source_bindings.private.json']=c.read('native_source_bindings.private.json')
    plan=seal(dict(schema='seek-output-contract-v2-proposal',version=candidate['version'],
        parent_packet_sha256=PARENT,parent_proposal_hash=json.loads(outer.read('candidate/proposal.private.json'))['hash'],
        previous_response_draft_hash='158ec1539a5bfc93a7ad13e321c31da2fba0e1943ac8baededa4c55da6b09798',
        base_count=len(base),base_hash=digest(base),base_annotations_hash=digest(anns),
        corpus_hashes={r:digest(candidate['corpora'][r]) for r in ('C','LQ')},
        annotation_hashes={r:digest(candidate['annotations'][r]) for r in ('C','LQ')},
        contract_hash=digest(contract),tasks_hash=digest(tasks),lineage_hash=digest(lineage),
        native_source_bindings_hash=digest(read('native_source_bindings.private.json')),
        neutral_thought=NEUTRAL_THOUGHT,supervision_policy=candidate['supervision_policy'],inference_contract=candidate['inference_contract'],
        native_masks_valid=False,old_native_arrays_usable_for_changed_additions=False,
        stage='development_proposal',optimizer_hyperparameters_changed=False,supervision_policy_changed=True,source_files=sources,
        transfer_attempts=132,optional_seen_context_enabled=False,optional_seen_context_attempts=32,
        source_approved=False,final_corpus_approved=False,training_authorized=False,execution_authorized=False,installation_authorized=False))
    add('plan.private.json',plan)
    summary=dict(status=candidate['status'],hash=plan['hash'],sources=len(slots),rows_per_role={r:len(candidate['corpora'][r]) for r in ('C','LQ')},
        proposed_responses=len(candidate['rows']),active_responses_revised=22,
        nonactive_original_responses_preserved=sum(x['original_response_preserved'] for x in candidate['rows']),
        parser_agreement=len(candidate['rows']),legal_benign_labels=sum(x['score']['legal_endpoint_value']==0 for x in candidate['rows']),
        legal_target_labels=sum(x['score']['legal_endpoint_value']==1 for x in candidate['rows']),
        native_masks_valid=False,model_calls=0,training_authorized=False,execution_authorized=False,installation_authorized=False)
    add('summary.json',summary)
    for name,b in files.items():
        with (out/name).open('xb') as f:f.write(b)
    entries=dict(files,**{'parent.review.private.zip':raw})
    for name in sources:entries['source/'+name]=(ROOT/name).read_bytes()
    receipt=write_zip(out/'review.private.zip',entries)
    (out/'bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.packet,a.output),indent=2))
