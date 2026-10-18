"""Query-only production runner; offline zero-inference attestation precedes approval."""
import argparse
import copy
import inspect
import json
import os
import sys
from pathlib import Path
from bind_query_continuation import ROOT, preflight
from query_only_guard import QueryOnlyVictim
from seek.discovery_phase import common as c, hacs as h
from seek.schemas import Invalid, digest
from seek.storage import immutable_json, read_json, row_lock
from seek.victim import LegacyVictim, legacy_reset_text, file_hash
from seek.provenance import audit_checkpoint


def sources():
    result=c.source()
    for name in ('run_scoped_query.py','query_only_guard.py','bind_query_continuation.py'):
        result['docs/seek/hacs_2026_10_13/'+name]=file_hash(Path(__file__).with_name(name))
    for name in ('victim.py','provenance.py','storage.py','schemas.py'):
        p=ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek'/name
        result[str(p.relative_to(ROOT))]=file_hash(p)
    test_path=ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/tests/seek/test_hacs_reference_review.py'
    result[str(test_path.relative_to(ROOT))]=file_hash(test_path)
    return result


def metadata_view(entry,generation):
    # No AutoModel import, from_pretrained model call or CUDA operation.
    import torch
    import transformers
    from transformers import AutoTokenizer
    from fastchat.model.model_adapter import get_conversation_template
    v=LegacyVictim.__new__(LegacyVictim)
    v.torch=torch
    v.backend_identity='transformers:'+transformers.__version__+';torch:'+torch.__version__
    v.make_conversation=get_conversation_template
    v.system,demonstration,v.source_hash=legacy_reset_text()
    v.initial_history=[dict(role='user',content=demonstration),dict(role='assistant',content='Ok.')]
    v.identity=entry['identity'];v.dtype=generation['dtype']
    v.tokenizer=AutoTokenizer.from_pretrained(entry['path'],local_files_only=True)
    if v.tokenizer.pad_token is None:v.tokenizer.pad_token=v.tokenizer.eos_token
    conv=v.make_conversation('llama-2');conv.set_system_message(v.system)
    v.template_hash=digest(dict(template=conv.dict(),class_source=inspect.getsource(type(conv))))
    names=('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json')
    v.tokenizer_meta=dict(identity=digest({n:file_hash(Path(entry['path'])/n) for n in names if (Path(entry['path'])/n).exists()}),
        truncation_side=v.tokenizer.truncation_side,padding_side=v.tokenizer.padding_side)
    return v


def serialization(draft,v):
    rows=[];generation=draft['profile']['generation']
    for pair in draft['cells']:
        for cell in pair:
            # Only policy_input, never the whole cell/request, reaches render.
            prompt=v.render(v.initial_history,cell['policy_input'])
            ids,full=v.encode(prompt,generation)
            if ids!=full:raise Invalid('truncated full serialized input')
            if draft['outcome']['target'].casefold() in prompt.casefold():raise Invalid('target leaked into full serialization')
            if cell['policy_input'] not in prompt:raise Invalid('policy input absent from serialization')
            rows.append(dict(cell_id=cell['id'],policy_input=cell['policy_input'],serialized_prompt=prompt,
                utf8_sha256=__import__('hashlib').sha256(prompt.encode('utf-8')).hexdigest(),
                encoded_ids=ids,full_ids=full,input_tokens=len(ids)))
    return dict(rows=rows,system=v.system,initial_history=v.initial_history,identity=v.identity,
        tokenizer=v.tokenizer_meta,template_hash=v.template_hash,backend=v.backend_identity,dtype=v.dtype)


def attest(draft,view_factory=metadata_view,auditor=audit_checkpoint):
    preflight(draft)
    if draft['profile']['generation']['do_sample'] is not False:raise Invalid('deterministic generation required')
    entry=draft['profile']['models']['query']
    inventory=auditor(entry['path'])
    if inventory['weights']!=entry['weights'] or inventory['identity']!=entry['identity']:
        raise Invalid('actual checkpoint inventory mismatch')
    if 'config.json' not in inventory['weights'] or not any(x.endswith('.safetensors') for x in inventory['weights']):
        raise Invalid('config and safetensors binding required')
    view=view_factory(entry,draft['profile']['generation'])
    rendered=serialization(draft,view)
    return c.seal(dict(schema='seek-query-production-attestation-v1',draft_hash=draft['hash'],
        source=sources(),query_entry_hash=draft['execution_target_entry_hash'],inventory=inventory,
        serialization=rendered,worker='query_only_guard.QueryOnlyVictim',
        call_limits=dict(query_probes=4,query_anchors=2,total=6,observation=0,reference=0,defender=0,retries=0),
        model_calls=0,weights_loaded=False,execution_authorized=False,
        scope='offline serialization/token/weight verification; no inference or semantic validation'))


def approved(draft,attestation,review):
    for x in (draft,attestation,review):c.checked(x)
    required=('production_guard_and_limits','serialization_tokens_and_weights','resources')
    if (review.get('schema')!='seek-query-execution-review-v1' or review.get('decision')!='approve'
        or review.get('independent') is not True or not review.get('reviewer')
        or review.get('execution_authorized') is not True or review.get('resource_approved') is not True
        or review.get('draft_hash')!=draft['hash'] or review.get('attestation_hash')!=attestation['hash']
        or any(review.get('checks',{}).get(k) is not True for k in required)):
        raise Invalid('explicit draft-and-attestation-bound execution approval required')
    if attestation['draft_hash']!=draft['hash'] or attestation['source']!=sources():raise Invalid('stale attestation')


def run(draft,attestation,review,out,loader=LegacyVictim,attester=attest):
    approved(draft,attestation,review)
    if not os.environ.get('SLURM_JOB_ID'):raise Invalid('Slurm allocation required')
    # Rehash weights and serialize with real tokenizer before loading any model.
    if attester(draft)!=attestation:raise Invalid('live preflight differs from approved attestation')
    out=Path(out)
    with row_lock(out):
        if (out/'started.json').exists() or (out/'victim').exists():raise Invalid('existing execution; no automatic resume')
        immutable_json(out/'draft.json',draft);immutable_json(out/'attestation.json',attestation)
        immutable_json(out/'approval.json',review);immutable_json(out/'started.json',dict(draft_hash=draft['hash']))
        victim=None;rows=[];failure=None;stable=None;pairs=[]
        try:
            victim=QueryOnlyVictim(draft,loader)
            if serialization(draft,victim)!=attestation['serialization']:raise Invalid('loaded model interface differs')
            generation=draft['profile']['generation']
            cells=[copy.deepcopy(x) for pair in draft['cells'] for x in pair]
            cells += [dict(copy.deepcopy(x),id='anchor-'+x['id'],anchor=True) for x in draft['anchor_cells']]
            for cell in cells:
                row=c.responses(victim,[cell],generation,out/'victim',draft['hash'],h.score)[0]
                rows.append(row)
                if row['score']['value'] is None:raise Invalid('unscorable response; stop without exclusion')
            stable=all(rows[i]['reply']['raw_response']==rows[4+i]['reply']['raw_response'] for i in range(2))
            if not stable:raise Invalid('unstable anchors; interpretation stopped')
            if victim.call_receipt()['total_attempts']!=6:raise Invalid('expected six physical attempts')
            for i in (0,2):
                values=[rows[i]['score']['value'],rows[i+1]['score']['value']]
                pairs.append(dict(values=values,difference=values[0]-values[1]))
        except Exception as exc:failure=dict(type=type(exc).__name__,reason=str(exc))
        finally:
            counts=None if victim is None else victim.call_receipt()
            if victim is not None:c.release(victim._victim)
        result=c.seal(dict(schema='seek-scoped-query-result-v1',draft_hash=draft['hash'],attestation_hash=attestation['hash'],
            status='backend_failure' if failure else 'development_complete',failure=failure,rows=rows,paired_results=pairs,
            anchors_stable=stable,guard_counts=counts,costs=c.costs(out/'victim'),development_reuse=True,
            simulated=None if victim is None else victim.simulated,
            scientific_confirmation=False,method_matrix_unlocked=False,registry_mutations=0,
            scope=draft['claim_scope']))
        immutable_json(out/'result.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['attest','run']);parser.add_argument('--draft',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--attestation');parser.add_argument('--approval')
    args=parser.parse_args(argv);draft=c.checked(read_json(args.draft))
    if args.command=='attest':
        result=attest(draft);immutable_json(Path(args.output)/'attestation.json',result)
        template=c.seal(dict(schema='seek-query-execution-review-v1',draft_hash=draft['hash'],attestation_hash=result['hash'],
            decision='pending',reviewer='',independent=None,resource_approved=False,execution_authorized=False,
            checks={k:None for k in ('production_guard_and_limits','serialization_tokens_and_weights','resources')},
            findings=[],method_matrix_unlocked=False))
        immutable_json(Path(args.output)/'execution_review_template.json',template)
        print(json.dumps(dict(status='attestation_prepared',hash=result['hash'],draft_hash=draft['hash'],
            input_tokens=[x['input_tokens'] for x in result['serialization']['rows']],model_calls=0,
            execution_authorized=False,output=args.output),indent=2))
    else:
        if not args.attestation or not args.approval:parser.error('--attestation and --approval required')
        result=run(draft,read_json(args.attestation),read_json(args.approval),args.output)
        print(json.dumps(result,indent=2))
        if result['failure']:return 2
    return 0


if __name__=='__main__':
    sys.exit(main())
