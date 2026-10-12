import copy
import html
import json
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import immutable_json, read_json, row_lock, Journal
from ..semantic_registry import Registry
from ..next_phase import plans
from ..victim import file_hash


def source():
    result = plans.source()
    for p in list(Path(__file__).parent.glob('*.py')) + [plans.ROOT/'seek_discovery.py', plans.ROOT/'seek_discovery.sh']:
        if p.exists(): result[str(p.relative_to(plans.ROOT))] = file_hash(p)
    return result


def seal(value):
    value = copy.deepcopy(value)
    value['hash'] = digest({k:v for k,v in value.items() if k != 'hash'})
    return value


def checked(value):
    if value['hash'] != seal(value)['hash']: raise Invalid('mutated artifact')
    return value


def current(value):
    checked(value)
    if value['source'] != source(): raise Invalid('frozen discovery execution source changed')


def fingerprints(cells):
    return {digest(c['policy_input']) for c in cells}


def allocate(reg, plan, review):
    current(plan)
    if 'cells' in plan:
        actual=fingerprints(plan['cells'])
    else:
        actual={f for i in plan['incidents'] if not i['development_reuse'] for f in i['fingerprints']}
    if set(plan['fingerprints'])!=actual:raise Invalid('support fingerprint inventory differs from plan')
    if plan.get('j'): raise Invalid('already allocated input')
    if review.get('plan_hash') != plan['hash'] or not all(review.get(k) is True for k in ('accepted','independent','resource_approved')) or not review.get('reviewer') or not review.get('reason'):
        raise Invalid('independent input/scope/resource review required')
    with row_lock(reg.root):
        ev=reg.events()
        families=[e['data'] for e in ev if e['kind']=='family']
        if families != [dict(study_id='semantic_v1',delta=.05,simulated=plan['simulated'])]: raise Invalid('authoritative study family required')
        if not plan['simulated']:
            baseline=read_json(plans.ROOT/'docs/seek/reports/SEEK_FINAL_EXPORT_2026-10-08.json')
            if not any(e['hash']==baseline['registry_head'] for e in ev): raise Invalid('historical lineage missing')
        for e in ev:
            if e['kind']=='discovery_frozen' and e['data']['draft']==plan['hash']:
                old=load(reg,e['data']['j'])
                if old['review']!=review: raise Invalid('review differs')
                return old
        seen={f for e in ev if e['kind'] in ('allocation','exposure') for f in e['data']['fingerprints']}
        if seen & set(plan['fingerprints']): raise Invalid('support/lineage already exposed or reserved')
        p=copy.deepcopy(plan);p['j']=1+max((e['data']['j'] for e in ev if e['kind']=='allocation'),default=0);p['review']=review;p=seal(p)
        reg._append('allocation',dict(j=p['j'],contract_hash=p['hash'],fingerprints=p['fingerprints'],claim_origin=p['origin'],protocol=p['schema']))
        immutable_json(reg.root/'discovery_contracts'/f"{p['j']:06d}.json",p)
        reg._append('discovery_frozen',dict(j=p['j'],hash=p['hash'],draft=plan['hash']))
        return p


def load(reg,j):
    if type(j) is not int or j < 1: raise Invalid('positive allocation required')
    p=checked(read_json(reg.root/'discovery_contracts'/f'{j:06d}.json'))
    if not any(e['kind']=='discovery_frozen' and e['data']['hash']==p['hash'] for e in reg.events()): raise Invalid('unfrozen allocation')
    return p


def publish(folder, value):
    folder=Path(folder);immutable_json(folder/'result.json',value)
    page='<html><meta charset="utf-8"><title>Seek discovery result</title><pre>'+html.escape(json.dumps(value,indent=2))+'</pre></html>'
    path=folder/'report.html'
    if not path.exists(): path.write_text(page)
    return value


def release(v):
    if hasattr(v,'close'): v.close()
    elif hasattr(v,'model'):
        import gc
        del v.model;gc.collect();v.torch.cuda.empty_cache()


def responses(v,cells,generation,out,binding,score,reg=None):
    """One resident victim, exact cache binding, no unknown-execution retries."""
    if generation['do_sample'] is not False: raise Invalid('stochastic responses cannot reuse deterministic cache')
    out=Path(out);journal=Journal(out/'journal');rows=[]
    for cell in cells:
        prompt=v.render(v.initial_history,cell['policy_input']);ids,full=v.encode(prompt,generation)
        if ids!=full: raise Invalid('truncated consumed input')
        if cell.get('query_target') and cell['query_target'].casefold() in prompt.casefold(): raise Invalid('target leakage in full consumed query')
        rt=dict(checkpoint_identity=v.identity,template_hash=v.template_hash,tokenizer=v.tokenizer_meta,dtype=v.dtype,backend=v.backend_identity,generation=generation,system=v.system,prefix_messages=v.initial_history)
        if not v.simulated:
            rt['hardware']=dict(device=v.torch.cuda.get_device_name(0),cuda=v.torch.version.cuda)
        key=dict(binding=binding,prompt=prompt,ids=ids,runtime=rt,anchor=cell.get('anchor',False))
        kh=digest(key)
        if journal.completed(kh) is None and any(e['kind']=='call_attempt' and e['data']['key']==kh for e in journal.records): raise Invalid('unknown previous execution; do not retry')
        if reg is not None:
            # Token/consumed-text exposure stays private, separate from role payloads.
            fp={digest(prompt),digest(ids)}
            with row_lock(reg.root):
                events=reg.events()
                for e in events:
                    if e['kind']=='exposure' and fp & set(e['data']['fingerprints']) and e['data'].get('evidence',{}).get('owner','').split(':')[0]!=binding.split(':')[0]:
                        raise Invalid('consumed prompt/token overlap with another stream')
                if not any(e['kind']=='exposure' and e['data'].get('evidence',{}).get('owner')==binding and set(e['data']['fingerprints'])==fp for e in events):
                    reg._append('exposure',dict(fingerprints=sorted(fp),evidence=dict(owner=binding),evidence_phase='private_discovery_or_evaluation'))
        reply=journal.call(key,'victim','discovery_phase',10000,lambda:v.propose({'runtime':rt},cell['policy_input'],generation))
        if reply['serialized_prompt']!=prompt or reply['encoded_ids']!=ids or reply['full_ids']!=full or bool(reply['simulated'])!=bool(v.simulated): raise Invalid('victim response binding mismatch')
        import re
        labels=list(cell.get('slot_to_action',{}).values())+list(cell.get('brand_map',{}).values())
        positions={label:[m.start() for m in re.finditer(re.escape(label),prompt,re.I)] for label in labels}
        r=dict(cell=cell,reply=reply,runtime=rt,consumed_positions=positions,score=score(reply['raw_response'],cell),key=kh)
        immutable_json(out/'rows'/f"{cell['id']}.json",r);rows.append(r)
    return rows


def costs(folder,journal_subdir='journal'):
    from ..storage import events
    es=events(Path(folder)/journal_subdir/'events.jsonl')
    replies=[e['data']['result'] for e in es if e['kind']=='call_complete']
    return dict(physical_calls=sum(e['kind']=='call_attempt' for e in es),cache_hits=sum(e['kind']=='cache_hit' for e in es),
                reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in replies),tokens_reported=any('usage' in r for r in replies))


def verify_rows(result,plan,folder,score):
    """Independent raw/journal and scorer reconstruction; does not repair outputs."""
    from ..storage import events
    es=events(Path(folder)/'journal/events.jsonl')
    for i,e in enumerate(es):
        if e['sequence']!=i or e['id']!=digest([i,e['kind'],e['data']]):raise Invalid('journal mutation')
    expected={x['id']:x for x in plan['cells']}
    for aid in plan.get('anchor_ids',[x['id'] for x in plan['cells'][:2]]):
        cell=copy.deepcopy(expected[aid]);cell.update(id='anchor-'+aid,anchor=True);expected[cell['id']]=cell
    if len(result['rows'])!=len(expected) or {r['cell']['id'] for r in result['rows']}!=set(expected):raise Invalid('missing/duplicate physical row records')
    for r in result['rows']:
        cell=r['cell'];reply=r['reply']
        if cell!=expected[cell['id']] or read_json(Path(folder)/'rows'/f"{cell['id']}.json")!=r:raise Invalid('row/plan mismatch')
        rt=r['runtime']
        if rt['checkpoint_identity']!=plan['profile']['models'][result['model']]['identity'] or rt['generation']!=plan['profile']['generation']:raise Invalid('runtime identity mismatch')
        key=dict(binding=plan['hash']+':'+result['model'],prompt=reply['serialized_prompt'],ids=reply['encoded_ids'],runtime=rt,anchor=cell.get('anchor',False))
        if digest(key)!=r['key'] or reply['encoded_ids']!=reply['full_ids'] or cell['policy_input'] not in reply['serialized_prompt']:raise Invalid('consumed binding mismatch')
        completions=[e for e in es if e['kind']=='call_complete' and e['data']['key']==r['key']]
        if len(completions)!=1 or completions[0]['data']['result']!=reply or score(reply['raw_response'],cell)!=r['score']:raise Invalid('response/journal/scorer mismatch')
    if costs(folder)!=result['costs']:raise Invalid('cost accounting mismatch')
