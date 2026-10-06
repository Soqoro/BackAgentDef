"""Offline, stateless paired execution; immutable draw/arm ledger and resume."""
import copy
import os
import random
import time
from pathlib import Path
from .schemas import Invalid,digest
from .storage import immutable_json,atomic_json,read_json,row_lock,events
from .semantic_stats import bounds,positive
from .semantic_renderers import render,score,audit_prompt


def draw(c,index):
    # Separate PRNGs for IID background indices and diagnostic arm order.
    rng=random.Random(c['sampling']['seed'])
    selected=None
    for _ in range(index+1): selected=rng.randrange(len(c['sampling']['pool']))
    order=['1','0']; random.Random(digest([c['sampling']['seed'],'arm-order',index])).shuffle(order)
    b=c['sampling']['pool'][selected]
    return dict(index=index,background=copy.deepcopy(b),support_index=selected,order=order,
                contract_hash=c['contract_hash'],claim_origin=c['origin'],evidence_phase='confirmation',
                arms=render(c['renderer']['spec'],b))


def prepare(victim,snapshot,c,d):
    out={}
    for arm in ('1','0'):
        r=d['arms'][arm]; snap=copy.deepcopy(snapshot)
        # Reset to original policy demonstration; do not retain incident exposure/history.
        snap['runtime']['prefix_messages']=copy.deepcopy(victim.initial_history)
        snap['runtime']['structured_state']=copy.deepcopy(r['state'])
        snap['runtime']['frozen_contract']={'instruction':r['instruction'],'provenance':'hypothetical_diagnostic','explicit_prohibitions':[]}
        prompt=victim.render(victim.initial_history,r['policy_input'])
        ids,full=victim.encode(prompt,snap['runtime']['generation'])
        if ids!=full: raise Invalid('pre_response_truncation; stream blocked, no resampling')
        out[arm]=dict(snapshot=snap,request=r['policy_input'],prompt=prompt,ids=ids,
                      input_audit=audit_prompt(prompt,c['renderer']['spec']),claim_origin=c['origin'],evidence_phase=d.get('evidence_phase','confirmation'))
    if c['renderer']['spec']['operator']!='slot_label' and any(p['input_audit']['brand'] for p in out.values()):
        raise Invalid('brand occurs in full consumed input; not unrequested insertion')
    return out


def check_policy(victim,snap,c):
    p=c['policy']; r=snap['runtime']
    if snap['hash']!=p['snapshot_hash'] or victim.identity!=p['identity'] or r['checkpoint_identity']!=victim.identity:
        raise Invalid('snapshot/checkpoint binding mismatch')
    if (digest(r['tokenizer'])!=p['tokenizer_hash'] or r['template_hash']!=p['template_hash'] or
        digest(r['generation'])!=p['generation_hash'] or victim.template_hash!=r['template_hash'] or victim.tokenizer_meta!=r['tokenizer']):
        raise Invalid('policy interface mismatch')
    if victim.simulated!=(c['origin']=='simulated'): raise Invalid('real/simulated population mismatch')


def run(registry,j,victim,snapshot,batch=None):
    c=registry.contract(j); check_policy(victim,snapshot,c)
    root=registry.root/('simulated' if victim.simulated else 'real')/f'claim-{j:06d}'
    with row_lock(root):
        inf=c['inference']; cap=inf['max_pairs']; batch=inf['batch_pairs'] if batch is None else positive(batch,'batch')
        previous=None; pairs=[]; groups=set(); prompts=set(); templates=set()
        failure=None; completed_before=len(list((root/'pairs').glob('*.json')))
        stop=min(cap,completed_before+batch)
        for i in range(stop):
            d=draw(c,i); pd=root/'draws'/f'{i:06d}'
            immutable_json(pd/'manifest.json',d)
            path=root/'pairs'/f'{i:06d}.json'
            if path.exists():
                record=read_json(path)
                if record['hash']!=digest({k:v for k,v in record.items() if k!='hash'}) or record['previous']!=previous or record['draw_hash']!=digest(d):
                    raise Invalid('paired ledger mutation or duplicate')
                for arm in ('1','0'):
                    if read_json(pd/f'arm{arm}.json')!=record['arms'][arm]: raise Invalid('arm ledger mismatch')
            else:
                try:
                    prepared=prepare(victim,snapshot,c,d)
                    # Inputs of BOTH arms are validated and persisted before any response.
                    immutable_json(pd/'inputs.json',{a:{k:v for k,v in p.items() if k!='snapshot'} for a,p in prepared.items()})
                    results={}
                    for arm in d['order']:
                        ap=pd/f'arm{arm}.json'; attempt=pd/f'attempt{arm}.json'
                        if ap.exists():
                            results[arm]=read_json(ap)
                            if victim.simulated and hasattr(victim,'restore') and 'simulator_state_after' in results[arm]['reply']:
                                victim.restore(results[arm]['reply']['simulator_state_after'])
                            continue
                        if attempt.exists(): raise Invalid('unresolved prior arm attempt; no automatic reroll')
                        if len(list((root/'draws').glob('*/attempt*.json')))>=c['costs']['max_victim_calls']:
                            raise Invalid('victim budget exhausted')
                        immutable_json(attempt,dict(draw=i,arm=arm,input_hash=digest(prepared[arm]['ids']),contract_hash=c['contract_hash']))
                        started=time.monotonic(); p=prepared[arm]
                        reply=victim.propose(p['snapshot'],p['request'],snapshot['runtime']['generation'])
                        if reply['encoded_ids']!=p['ids'] or reply['full_ids']!=p['ids'] or reply['simulated']!=victim.simulated:
                            raise Invalid('response identity/truncation mismatch')
                        results[arm]=dict(claim_origin=c['origin'],evidence_phase='confirmation',contract_hash=c['contract_hash'],reply=reply,score=score(reply['raw_response'],c['renderer']['spec']),
                                          latency_seconds=time.monotonic()-started,input_audit=p['input_audit'])
                        immutable_json(ap,results[arm])
                    if any(r['score']['value'] is None for r in results.values()):
                        raise Invalid('unscorable pair; fail_closed, no exclusion')
                    record=dict(index=i,previous=previous,contract_hash=c['contract_hash'],draw_hash=digest(d),
                                claim_origin=c['origin'],evidence_phase='confirmation',arms=results)
                    record['hash']=digest(record); immutable_json(path,record)
                except Exception as exc:
                    failure={'index':i,'error_type':type(exc).__name__,'reason':str(exc) if isinstance(exc,Invalid) else 'backend exception; inspect stderr'}
                    immutable_json(pd/('failure-'+digest(failure)+'.json'),failure)
                    break
            for arm_record in record['arms'].values():
                if arm_record.get('claim_origin')!=c['origin'] or arm_record.get('evidence_phase')!='confirmation' or arm_record.get('contract_hash')!=c['contract_hash']:
                    raise Invalid('arm provenance mismatch')
            if victim.simulated and hasattr(victim,'restore'):
                last=record['arms'][d['order'][-1]]['reply'].get('simulator_state_after')
                if last is not None: victim.restore(last)
            previous=record['hash']; pairs.append([record['arms']['1']['score']['value'],record['arms']['0']['score']['value']])
            groups.add(d['background']['group']); templates.add(d['background']['phrasing'])
            prompts.update(digest(r['reply']['encoded_ids']) for r in record['arms'].values())
            estimate=bounds(pairs,j,inf['delta'],inf['tau'],inf['semantic_eta'])
            immutable_json(root/'bounds'/f'{i:06d}.json',dict(pair_hash=previous,**estimate))
            if estimate['implemented_certified'] and (inf['semantic_eta'] is None or estimate['semantic_certified_conditional']): break
        estimate=bounds(pairs,j,inf['delta'],inf['tau'],inf['semantic_eta'])
        terminal=bool(estimate['implemented_certified'] and (inf['semantic_eta'] is None or estimate['semantic_certified_conditional'])) or len(pairs)>=cap
        inference=('inference_invalid' if failure else 'semantic_effect_certified_conditional' if estimate['semantic_certified_conditional'] else
                   'implemented_effect_certified' if estimate['implemented_certified'] else 'inconclusive' if terminal else 'confirming')
        # Do not expose a positive flag on a prefix when an unresolved pair blocks the stream.
        if failure:
            estimate['implemented_certified']=False
            estimate['semantic_certified_conditional']=None if inf['semantic_eta'] is None else False
        result=dict(schema_version='semantic-result-v1',j=j,contract_hash=c['contract_hash'],claim_origin=c['origin'],
                    evidence_phase='confirmation',simulated=victim.simulated,execution='backend_failure' if failure else 'completed' if terminal else 'running',
                    inference=inference,semantic_support=inf['semantic_support'],bounds=estimate,failure=failure,
                    costs=dict(victim_attempts=len(list((root/'draws').glob('*/attempt*.json'))),
                               victim_responses=len(list((root/'draws').glob('*/arm*.json'))),replays=sum(e['kind']=='call_attempt' for e in events(root/'replay'/'events.jsonl')),cache_hits=0,defender_calls=0),
                    diversity=dict(unique_groups=len(groups),unique_prompts=len(prompts),finite_support=len(c['sampling']['pool']),unique_templates=len(templates),
                                   unique_underlying_catalogue_products=None,group_semantics='constructed neutral backgrounds; not independently trained attacks'),
                    next_pair_index=len(pairs),ledger_head=previous,unconditional_semantic_confirmation=False,
                    scope=c['behavior']['scope'],slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        atomic_json(root/'result.json',result)
        registry.append('status',dict(j=j,result_hash=digest(result),claim_origin=c['origin'],evidence_phase='confirmation',
                                     execution=result['execution'],inference=inference))
        return result
