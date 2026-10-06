"""Semantic Seek recipes. All real inference requires explicit Slurm allocation."""
import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path
from .schemas import Invalid,digest
from .storage import immutable_json,read_json,Journal,row_lock
from .snapshot_io import validate_snapshot
from .semantic_registry import Registry
from .semantic_contracts import draft,validate_contract,REVIEW,review_target
from .semantic_renderers import render,support,score
from .semantic_runner import prepare,check_policy,run
from .semantic_simulation import pool,policy_binding,study
from .semantic_evidence import import_evidence,export,aggregate,fingerprints

ROOT=Path(__file__).resolve().parents[4]


def source_hash():
    from .cli import source_metadata
    from .victim import file_hash
    return digest(dict(core=source_metadata(),semantic_entry=file_hash(ROOT/'seek_semantic.py'),worker=file_hash(ROOT/'seek_semantic.sh')))


def inputs(config_path,snapshot_path):
    from .provenance import checkpoint_entry
    config=read_json(config_path); snap=read_json(snapshot_path); validate_snapshot(snap)
    entry=checkpoint_entry(config,{'checkpoint_alias':snap['public']['checkpoint_alias']})
    if config['simulated'] or snap['public']['simulated']: raise Invalid('real recipe requires real snapshot/config')
    if not entry['enabled'] or entry['identity']!=snap['runtime']['checkpoint_identity'] or digest(entry['weights'])!=entry['identity']:
        raise Invalid('checkpoint registry identity mismatch or disabled entry')
    if config['victim']!=snap['runtime']['generation']: raise Invalid('config changes captured victim generation')
    binding=policy_binding(snap,digest(entry),source_hash())
    return config,snap,entry,binding


def real_victim(entry,snap):
    if not os.environ.get('SLURM_JOB_ID'): raise Invalid('real generation requires a Slurm allocation')
    from .provenance import verify_weights
    from .victim import LegacyVictim
    verify_weights(entry)
    from .manifests import resolve
    return LegacyVictim(dict(entry,path=str(resolve(entry['path']))),snap["runtime"]["generation"])


def release(victim):
    torch=victim.torch
    del victim.model
    gc.collect(); torch.cuda.empty_cache()


def proposed(args,spec,binding,snap,start=None):
    c=draft(args.study_id,binding,spec,pool(args.pool_start if start is None else start,args.pool_size),
            origin='incident_led' if args.command in ('role-smoke','discover') else 'evaluator_specified',seed=args.seed,
            method=getattr(args,'method','adaptive'))
    c['evidence']['incident_ids']=[snap['public']['case_id']]
    c['evidence']['source_ids']=[digest(s) for s in snap['public']['sources']]
    return c


def replay_gate(victim,snap,journal):
    from .replay import no_edit_replay
    r=no_edit_replay(victim,snap,journal,budget=1)
    if r['status']!='replay_valid' or not r['raw_answer_equal'] or r['context_truncated']:
        raise Invalid('exact original replay failed')
    return r


def pilot_pairs(victim,snap,c,backgrounds,output,registry):
    check_policy(victim,snap,c); output=Path(output); completed=[]
    for i,b in enumerate(backgrounds):
        d=dict(index=i,background=b,evidence_phase='exploration',arms=render(c['renderer']['spec'],b),order=['1','0'])
        import random
        random.Random(digest([c['sampling']['seed'],'pilot-order',i])).shuffle(d['order'])
        ps=prepare(victim,snap,c,d)
        pd=output/f'pair-{i:03d}'
        immutable_json(pd/'inputs.json',{a:{k:v for k,v in p.items() if k!='snapshot'} for a,p in ps.items()})
        results={}
        for a in d['order']:
            ap=pd/f'arm{a}.json'; at=pd/f'attempt{a}.json'
            if ap.exists(): results[a]=read_json(ap); continue
            if at.exists(): raise Invalid('unresolved pilot arm; no reroll')
            immutable_json(at,dict(arm=a,input_hash=digest(ps[a]['ids'])))
            raw=victim.propose(ps[a]['snapshot'],ps[a]['request'],snap['runtime']['generation'])
            if raw['encoded_ids']!=ps[a]['ids'] or raw['full_ids']!=ps[a]['ids'] or raw['simulated']!=victim.simulated:
                raise Invalid('pilot input mismatch')
            results[a]=dict(claim_origin=c['origin'],evidence_phase='exploration',draft_hash=review_target(c),reply=raw,score=score(raw['raw_response'],c['renderer']['spec']))
            immutable_json(ap,results[a])
        rec=dict(id=digest([c['renderer']['spec'],b]),background=b,arms=results,claim_origin=c['origin'],evidence_phase='exploration')
        immutable_json(pd/'result.json',rec); completed.append(rec)
        registry.expose([*support(c['renderer']['spec'],[b]),b['group']],dict(probe_id=rec['id']))
        if any(r['score']['value'] is None for r in results.values()): raise Invalid('unscorable exploration pair')
    return completed


def discovery(args):
    from .semantic_roles import SemanticRoles,discuss,select_probe
    config,snap,entry,binding=inputs(args.config,args.snapshot)
    if not os.environ.get('SLURM_JOB_ID'): raise Invalid('real roles require Slurm')
    out=Path(args.output); reg=Registry(args.registry); reg.expose(fingerprints(snap),{'snapshot_hash':snap['hash']})
    rounds=1 if args.command=='role-smoke' else 6
    with row_lock(out):
        immutable_json(out/'manifest.json',dict(source_hash=source_hash(),snapshot_hash=snap['hash'],binding=binding,
                       logs={'stdout':f"logs/seek/semantic-{os.environ.get('SLURM_JOB_ID')}.out",'stderr':f"logs/seek/semantic-{os.environ.get('SLURM_JOB_ID')}.err"},method=args.method,origin='incident_led',max_rounds=rounds,max_blocks=12,max_victim_calls=64,
                       max_logical_roles=24,retries_per_role=2,slurm_job_id=os.environ.get('SLURM_JOB_ID'),arguments=vars(args)))
        journal=Journal(out); evidence=[]; choices=[]; transcripts=[]; startup=[]
        result=dict(execution='running',inference='exploratory',simulated=False,claim_origin='incident_led',scientific_confirmation=False)
        try:
            for r in range(rounds):
                rp=out/f'round-{r:02d}'; backend=SemanticRoles(config['agents']); started=time.monotonic()
                try: transcript=discuss(backend,snap,journal,rp/'discussion.json',evidence,r,args.method,choices)
                finally: backend.close()
                startup.append(dict(phase='defender_including_discussion',seconds=time.monotonic()-started))
                transcripts.append(transcript); last=transcript['replies'][-1]
                chosen=transcript['selected_candidate']
                spec=last['spec']
                c=proposed(args,spec,binding,snap)
                c['condition']['definition']=last['condition'] if spec==last['spec'] else c['condition']['definition']; c.update(alternatives=last['alternatives'],predictions=last['predictions'],rationale=last['rationale'])
                c['evidence']['probe_ids']=[x['id'] for x in evidence]
                immutable_json(rp/'draft.json',c)
                if chosen is None: break
                started=time.monotonic(); victim=real_victim(entry,snap)
                startup.append(dict(phase='victim_load',seconds=time.monotonic()-started))
                try:
                    replay_gate(victim,snap,Journal(rp/'replay'))
                    # Discovery support is disjoint from the proposed confirmation budget range.
                    probes=pilot_pairs(victim,snap,c,pool((20 if args.command=='role-smoke' else 40)+r*2,2),rp/'probes',reg)
                finally: release(victim)
                choices.append(dict(candidate_hash=digest(chosen),operator=spec['operator'],
                                    mean_difference=sum(p['arms']['1']['score']['value']-p['arms']['0']['score']['value'] for p in probes)/len(probes)))
                evidence.extend({'id':p['id'],'scores':{a:v['score'] for a,v in p['arms'].items()},'spec':spec} for p in probes)
                journal.emit('semantic_evidence_update',dict(round=r,probe_ids=[p['id'] for p in probes],selected_candidate=chosen))
                immutable_json(rp/'completed.json',dict(status='completed_nonempty_pair_cycle',probe_ids=[p['id'] for p in probes],draft_hash=digest(c)))
            result.update(execution='completed',status='candidate_only' if not evidence else 'completed_nonempty_investigation',
                          completed_pairs=len(evidence),defender_logical_calls=sum(x['kind']=='semantic_logical_role' for x in journal.records),
                          defender_attempts=sum(x['kind']=='call_attempt' for x in journal.records),
                          startup_and_phase_times=startup,next_step='Independent review and register a fresh confirmation contract.')
        except Exception as exc:
            if isinstance(exc,Invalid) and 'library exhausted' in str(exc) and evidence:
                result.update(execution='completed',status='completed_nonempty_investigation',completed_pairs=len(evidence),stop_reason='finite_library_exhausted')
            else:
                result.update(execution='backend_failure',reason=str(exc) if isinstance(exc,Invalid) else type(exc).__name__)
        from .storage import atomic_json
        result['victim_attempts']=len(list(out.glob('round-*/probes/pair-*/attempt*.json')))+sum(
            sum(e['kind']=='call_attempt' for e in Journal(p).records) for p in out.glob('round-*/replay'))
        result.update(method=args.method,opened_incidents=1,defender_logical_calls=sum(x['kind']=='semantic_logical_role' for x in journal.records),
                      defender_attempts=sum(x['kind']=='call_attempt' for x in journal.records))
        atomic_json(out/'result.json',result)
        reg.append('investigation',dict(investigation_id=digest(str(out.resolve())),result=result))
        return result


def parser():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('budget')
    for name,typ,default in [('j',int,1),('delta',float,.05),('tau',float,.2),('eta',float,None)]: b.add_argument('--'+name,type=typ,default=default)
    for name in ('import-status','status','export'):
        s=sub.add_parser(name); s.add_argument('--registry',required=True)
        if name=='import-status':
            s.add_argument('--native-root',default='results/seek/diagnostics/native_characterization_v1'); s.add_argument('--historical-root',action='append',default=[])
        if name=='export': s.add_argument('--output',required=True)
    s=sub.add_parser('simulate'); s.add_argument('--output',required=True); s.add_argument('--replications',type=int,default=2)
    s.add_argument('--max-pairs',type=int,default=64); s.add_argument('--seed',type=int,default=123); s.add_argument('--rules',nargs='+',default=['null','concept'])
    for name in ('compile','role-smoke','discover','observation-pilot'):
        s=sub.add_parser(name)
        for field in ('config','snapshot','output','study-id'): s.add_argument('--'+field,required=True)
        s.add_argument('--pool-start',type=int,default=100); s.add_argument('--pool-size',type=int,default=64); s.add_argument('--seed',type=int,default=42)
        if name in ('compile','observation-pilot'): s.add_argument('--spec',required=True)
        if name!='compile': s.add_argument('--registry',required=True)
        if name in ('role-smoke','discover'): s.add_argument('--method',choices=['adaptive','fixed','discussion_only'],default='adaptive')
        if name=='observation-pilot': s.add_argument('--review',required=True)
    s=sub.add_parser('preview'); s.add_argument('--draft',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('review-template'); s.add_argument('--draft',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('review'); s.add_argument('--draft',required=True); s.add_argument('--review',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('register'); s.add_argument('--draft',required=True); s.add_argument('--registry',required=True)
    s=sub.add_parser('confirm'); s.add_argument('--registry',required=True); s.add_argument('--j',type=int,required=True)
    s.add_argument('--config',required=True); s.add_argument('--snapshot',required=True); s.add_argument('--batch',type=int)
    s=sub.add_parser('audit-opportunities'); s.add_argument('--run-root',required=True); s.add_argument('--output',required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        cmd=args.command
        if cmd=='budget':
            from .semantic_stats import budget
            result=budget(args.j,args.delta,args.tau,args.eta)
        elif cmd=='import-status': result=import_evidence(Registry(args.registry),args.native_root,args.historical_root)
        elif cmd=='status': result=aggregate(Registry(args.registry))
        elif cmd=='export': result=export(Registry(args.registry),args.output)
        elif cmd=='simulate': result=study(args.output,args.replications,args.max_pairs,args.rules,args.seed)
        elif cmd in ('role-smoke','discover'): result=discovery(args)
        elif cmd=='compile':
            config,snap,entry,binding=inputs(args.config,args.snapshot)
            result=proposed(args,read_json(args.spec),binding,snap); immutable_json(args.output,result)
        elif cmd=='preview':
            c=read_json(args.draft); validate_contract(c,False)
            result=dict(draft_hash=review_target(c),model_calls=0,claim_origin=c['origin'],evidence_phase='design',
                        pairs=[dict(background=b,rendered=render(c['renderer']['spec'],b)) for b in c['sampling']['pool']])
            immutable_json(args.output,result)
        elif cmd=='review-template':
            c=read_json(args.draft); validate_contract(c,False)
            result=dict(draft_hash=review_target(c),accepted=False,reviewer='',independent=True,reason='pending independent review of paired inputs and claim scope',evidence_ids=c['evidence']['incident_ids'])
            immutable_json(args.output,result)
        elif cmd=='review':
            from .schemas import validate
            result=read_json(args.draft); validate_contract(result,False)
            if result['j']!=0 or result['contract_hash']: raise Invalid('cannot revise frozen contract')
            review=read_json(args.review); validate(review,REVIEW)
            result['renderer']['review']=review
            if review['draft_hash']!=review_target(result): raise Invalid('review draft hash mismatch')
            validate_contract(result,False); immutable_json(args.output,result)
        elif cmd=='register': result=Registry(args.registry).register(read_json(args.draft))
        elif cmd=='confirm':
            reg=Registry(args.registry); c=reg.contract(args.j)
            config,snap,entry,binding=inputs(args.config,args.snapshot)
            if binding!=c['policy']: raise Invalid('source/loader/policy binding changed; new contract required')
            # No Qwen process is started by confirmation.
            started=time.monotonic(); victim=real_victim(entry,snap)
            load_seconds=time.monotonic()-started
            try:
                root=reg.root/'real'/f'claim-{args.j:06d}'
                with row_lock(root/'replay'):
                    replay_gate(victim,snap,Journal(root/'replay'))
                immutable_json(root/'startup'/f"{os.environ['SLURM_JOB_ID']}.json",dict(seconds=load_seconds,job_id=os.environ['SLURM_JOB_ID'],source_hash=source_hash()))
                result=run(reg,args.j,victim,snap,args.batch)
            finally: release(victim)
        elif cmd=='observation-pilot':
            from .schemas import validate
            config,snap,entry,binding=inputs(args.config,args.snapshot); spec=read_json(args.spec)
            if spec['operator']!='slot_label': raise Invalid('observation pilot requires persistent slot operator')
            c=proposed(args,spec,binding,snap); review=read_json(args.review); validate(review,REVIEW)
            if review['draft_hash']!=review_target(c): raise Invalid('observation review draft hash mismatch')
            if not review['accepted'] or not review['independent']: raise Invalid('eight backgrounds need independent review')
            c['renderer']['review']=review
            if len(c['sampling']['pool'])<8: raise Invalid('observation pilot requires eight distinct backgrounds')
            out=Path(args.output); reg=Registry(args.registry)
            with row_lock(out):
                immutable_json(out/'draft.json',c)
                victim=real_victim(entry,snap)
                try:
                    replay=replay_gate(victim,snap,Journal(out/'replay'))
                    records=pilot_pairs(victim,snap,c,c['sampling']['pool'][:8],out/'probes',reg)
                finally: release(victim)
                result=dict(execution='completed',inference='exploratory',simulated=False,claim_origin='evaluator_specified',
                            evidence_phase='exploration',records=records,replay=replay,scientific_confirmation=False,
                            victim_calls=2*len(records)+1,scope='hypothetical persistent-slot inspection')
                immutable_json(out/'result.json',result)
        elif cmd=='audit-opportunities':
            from .characterization import opportunity_audit
            snaps=[read_json(p) for p in Path(args.run_root).glob('**/snapshots/*.json')]
            result=opportunity_audit(snaps); immutable_json(args.output,result)
        print(json.dumps(result,indent=2))
        return 2 if result.get('execution') in ('backend_failure','prerequisite_missing') else 0
    except (Invalid,OSError,ValueError,KeyError) as exc:
        failure=dict(execution='prerequisite_missing' if isinstance(exc,FileNotFoundError) else 'failed',
                     reason=str(exc),scientific_confirmation=False,command=args.command)
        if args.command in ('confirm','observation-pilot','discover','role-smoke'):
            destination=Path(args.output) if hasattr(args,'output') else Path(args.registry)/'real'/f'claim-{args.j:06d}'
            immutable_json(destination/'worker_failures'/(digest(failure)+'.json'),failure)
        print(json.dumps(failure),flush=True)
        return 2
