"""Metadata-only reconciliation and exports; missing artifacts remain unresolved."""
import html
from pathlib import Path
from .schemas import digest,Invalid
from .storage import read_json,immutable_json
from .snapshot_io import validate_snapshot

FINGERPRINTS={'task_fingerprint','instruction_fingerprint','trajectory_fingerprint','dependence_group','product_id','asin','group'}

def fingerprints(value):
    out=set()
    if isinstance(value,dict):
        for k,v in value.items():
            if k in FINGERPRINTS and isinstance(v,str): out.add(v)
            if k in ('instruction','policy_input','raw_request') and isinstance(v,str): out.add(digest(v))
            out.update(fingerprints(v))
    elif isinstance(value,list):
        for v in value: out.update(fingerprints(v))
    return out


def import_evidence(registry,native_root,historical_roots=()):
    root=Path(native_root); plan_path=root/'plan.json'
    result=dict(native_status='pending_missing',live_cluster_status=None,model_calls=0,
                evidence_phase='exploration',plan_hash=None,rows=[],base_reference=dict(kind='base_unpoisoned_reference',path=None),
                owner_report='user reports two trained query/observation mechanisms',training_binding='unverified')
    if plan_path.is_file():
        plan=read_json(plan_path)
        if plan.get('hash')!=digest({k:v for k,v in plan.items() if k!='hash'}): raise Invalid('native plan hash mismatch')
        result['plan_hash']=plan['hash']; result['native_status']='plan_received_outputs_pending'
        for i,row in enumerate(plan['rows']):
            p=root/f'row-{i:04d}'; selected=len(plan['tasks'])
            r=dict(row=i,checkpoint_alias=row['checkpoint_alias'],selected=selected,replay_valid=None,scorable=None,status='pending_missing')
            if (p/'result.json').is_file():
                out=read_json(p/'result.json'); manifest=read_json(p/'manifest.json')
                if manifest['plan']!=plan or manifest['row']!=i or out.get('checkpoint_alias')!=row['checkpoint_alias'] or out.get('simulated')!=plan['simulated']:
                    raise Invalid('native result/plan identity mismatch')
                snaps={}
                for sp in (p/'snapshots').glob('*.json'):
                    s=read_json(sp); validate_snapshot(s)
                    if s['runtime']['checkpoint_identity']!=row['checkpoint_identity']: raise Invalid('native checkpoint identity mismatch')
                    snaps[s['hash']]=s
                records=out.get('records',[])
                if any(x['snapshot_hash'] not in snaps for x in records): raise Invalid('native record missing bound snapshot')
                if len({x['snapshot_hash'] for x in records})!=len(records): raise Invalid('duplicate native records')
                r.update(status=out['status'],replay_valid=sum(x['status']=='replay_valid' for x in records),
                         scorable=sum(x['status']=='replay_valid' and x['measurements']['adidas_search'] is not None for x in records))
                if r['status']=='completed' and len(records)!=selected: raise Invalid('native selected denominator mismatch')
            result['rows'].append(r)
        if all(r['status']=='completed' for r in result['rows']): result['native_status']='received_descriptive_only'
    imported=[]
    for folder in (root,*map(Path,historical_roots)):
        for p in sorted(folder.rglob('*.json')) if folder.exists() else []:
            # Never walk a huge inventory; existing summaries/snapshots suffice.
            if 'inventory' in p.name: continue
            value=read_json(p); ids=fingerprints(value)
            if ids: registry.expose(ids,dict(artifact_hash=digest(value),path=str(p)))
            imported.append(dict(path=str(p),hash=digest(value),fingerprints=len(ids)))
    result['imported']=imported
    registry.append('evidence_import',result)
    return result


def verify_result(root,c,r):
    """Recompute raw-pair scores and confidence bounds on read-only export."""
    from .semantic_runner import draw
    from .semantic_renderers import score
    from .semantic_stats import bounds
    previous=None; pairs=[]
    paths=sorted((root/'pairs').glob('*.json'))
    for i,p in enumerate(paths):
        record=read_json(p); d=draw(c,i)
        if record['hash']!=digest({k:v for k,v in record.items() if k!='hash'}) or record['previous']!=previous or record['index']!=i or record['draw_hash']!=digest(d):
            raise Invalid('export pair ledger corruption')
        if read_json(root/'draws'/f'{i:06d}'/'manifest.json')!=d: raise Invalid('export draw mutation')
        for a in ('1','0'):
            arm=record['arms'][a]
            if arm['score']!=score(arm['reply']['raw_response'],c['renderer']['spec']): raise Invalid('scorer reconstruction mismatch')
        pairs.append([record['arms']['1']['score']['value'],record['arms']['0']['score']['value']])
        previous=record['hash']
    inf=c['inference']; expected=bounds(pairs,c['j'],inf['delta'],inf['tau'],inf['semantic_eta'])
    if r['failure']:
        expected['implemented_certified']=False
        expected['semantic_certified_conditional']=None if inf['semantic_eta'] is None else False
    if r['bounds']!=expected or r['ledger_head']!=previous or r['next_pair_index']!=len(pairs):
        raise Invalid('summary does not reproduce from raw pairs')


def aggregate(registry):
    claims=[]; population=set()
    for p in sorted((registry.root/'contracts').glob('*.json')):
        c=registry.contract(int(p.stem)); simulated=c['origin']=='simulated'; population.add(simulated)
        rp=registry.root/('simulated' if simulated else 'real')/f"claim-{c['j']:06d}"/'result.json'
        r=read_json(rp) if rp.exists() else None
        if r and (r['contract_hash']!=c['contract_hash'] or r['simulated']!=simulated): raise Invalid('result contract/population mismatch')
        if r: verify_result(rp.parent,c,r)
        claims.append(dict(contract=c,result=r,execution='planned' if r is None else r['execution']))
    if len(population)>1: raise Invalid('mixed real/simulated aggregation prohibited')
    return dict(schema_version='semantic-export-v1',simulated=next(iter(population),None),claims=claims,
                registry_head=registry.events()[-1]['hash'] if registry.events() else None,
                historical_shield_metrics=None,malicious_training_attribution=None,exact_trigger_recovery=None,
                investigations=list({r['data']['investigation_id']:r['data']['result'] for r in registry.events() if r['kind']=='investigation'}.values()),
                native_imports=[r['data'] for r in registry.events() if r['kind']=='evidence_import'])


def export(registry,output):
    report=aggregate(registry); output=Path(output); immutable_json(output/'report.json',report)
    import json
    body='<html><meta charset="utf-8"><title>Seek semantic audit</title><h1>Seek semantic audit</h1><p>Offline proposals only. Semantic bounds are conditional; null denotes unmeasured.</p><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></html>'
    path=output/'report.html'
    if path.exists() and path.read_text()!=body: raise Invalid('immutable HTML export collision; choose a new export directory')
    path.write_text(body)
    return report
