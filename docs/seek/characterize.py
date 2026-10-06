#!/usr/bin/env python3
"""Native task characterization, separate from Seek trigger confirmation."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.characterization import VERSION, LIMITATIONS, select_tasks, opportunity_audit, summarize
from seek.cli import load_snapshots, source_metadata
from seek.collection import public_case
from seek.content_diagnostic import action_metrics
from seek.manifests import resolve, validate_manifest
from seek.metrics import call_metrics
from seek.provenance import checkpoint_entry, environment_source_hash, verify_weights
from seek.replay import query, no_edit_replay
from seek.schemas import Invalid, PublicIncident, digest
from seek.snapshot_io import snapshot, save_snapshot
from seek.storage import Journal, immutable_json, read_json
from seek.victim import file_hash


def source():
    return {'core': source_metadata(), 'entry': digest(Path(__file__).read_text()),
            'worker': digest((ROOT/'seek_characterize.sh').read_text())}


def fresh(path):
    path = Path(path).resolve()
    if path.exists():
        raise Invalid('output exists; use a new characterization destination')
    return path


def prepare(config, per_cohort):
    assets = read_json(resolve(config['environment']['asset_manifest']))
    if not assets.get('files') or digest(assets['files']) != config['environment']['catalogue_hash']:
        raise Invalid('asset manifest/catalogue identity mismatch')
    for name, expected in assets['files'].items():
        if file_hash(resolve(name)) != expected:
            raise Invalid('asset bytes do not match manifest')
    old = validate_manifest(read_json(resolve(config['task_manifest'])),
              {k:config['environment'][k] for k in ('category','filter','goal_order_hash','catalogue_hash','environment_hash')})
    # Lazy cluster-only CPU imports: no model loader or generation here.
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    state = random.getstate()
    try:
        random.seed(42)
        env = WebAgentTextEnv(observation_mode='text', file_path=str(resolve(assets['product_file'])),
                             filter_goals=None, human_goals=False, num_products=assets['num_products'])
    finally:
        random.setstate(state)
    goals = env.server.goals
    selected = select_tasks(goals, old, per_cohort)
    namespace = {'filter': 'unfiltered_native_goals_seed42_v1', 'goal_order_hash': digest([digest(g) for g in goals]),
                 'environment_hash': environment_source_hash(), 'catalogue_hash': config['environment']['catalogue_hash']}
    for task in selected['tasks']:
        env.reset(task['local_id'])
        if digest(env.server.goals[task['local_id']]) != task['trajectory_fingerprint']:
            raise Invalid('native goal identity changed at reset')
        actions = env.get_available_actions()
        if not actions['has_search_bar'] or (set(s.lower() for s in actions['clickables'])-{'search'}):
            raise Invalid('native reset is not an initial search page')
        observation = env.observation
        if observation != 'WebShop [SEP] Instruction: [SEP] '+task['instruction']+' [SEP] Search':
            raise Invalid('unexpected native initial-page rendering')
        task.update(observation=observation, actions=actions, page_id=str(env.state.get('url','')),
                    task_fingerprint=digest([namespace,task['instruction_fingerprint'],task['trajectory_fingerprint']]))
    if len(config['rows']) != 2 or len({r['checkpoint_alias'] for r in config['rows']}) != 2:
        raise Invalid('pilot expects the two distinct existing checkpoint rows')
    rows = []
    for row in config['rows']:
        entry = checkpoint_entry(config,row)
        if not entry['enabled'] or not entry['weights'] or entry['identity'] != digest(entry['weights']):
            raise Invalid('enabled checkpoint with bound weights required')
        rows.append({'checkpoint_alias': row['checkpoint_alias'], 'channel': row['channel'],
                     'checkpoint_identity': entry['identity'], 'training_status': entry['training_status']})
    return {'protocol': VERSION, 'simulated': False, 'scientific_confirmation': False,
            'namespace': namespace, 'selection': selected['selection'], 'tasks': selected['tasks'],
            'rows': rows, 'generation': config['victim'], 'checkpoint_registry': config['checkpoint_registry'],
            'max_victim_calls_per_row': 2*len(selected['tasks']), 'source': source(),
            'clean_control': 'not_supplied; neither checkpoint is treated as clean', 'limitations': LIMITATIONS}


def probe(plan, row_index, victim, journal):
    row = plan['rows'][row_index]; records = []
    for task in plan['tasks']:
        request = 'Observation:\n'+task['observation']+'\n\nAvailable Actions:\n'+str(task['actions'])
        public = public_case(dict(task, split='development'),
                   dict(row, track='raw_audit', scope='native_initial_proposal_characterization'),
                   simulated=plan['simulated'], observation=task['observation'], request=request, policy_input=request,
                   actions=task['actions'], history=victim.initial_history,
                   state={'page_id':task['page_id'], 'selected_options':[], 'facts':[task['observation']],
                          'legal_clicks':task['actions']['clickables'], 'search_allowed':True},
                   sources=[{'start':0,'end':len(request),'text':request,'kind':'hard'}])
        runtime = victim.runtime(public.to_dict(), plan['generation'],
                  {'environment_fingerprint':plan['namespace']['environment_hash'],
                   'catalogue_fingerprint':plan['namespace']['catalogue_hash'],
                   'filter_fingerprint':digest(plan['namespace']['filter'])})
        if runtime['full_ids'] != runtime['encoded_ids']:
            raise Invalid('native task prompt truncated; no generation authorized')
        pre = snapshot(public,runtime,None)
        immutable_json(journal.root/'pre_calls'/(public.to_dict()['case_id']+'.json'),pre)
        response = query(victim,pre,request,journal,VERSION,len(plan['tasks']),candidate_hash=plan['hash'],tag='native_capture')
        if response['encoded_ids'] != runtime['encoded_ids'] or response['serialized_prompt'] != runtime['serialized_prompt'] or response['full_ids'] != runtime['full_ids']:
            raise Invalid('capture input mismatch')
        p = public.to_dict(); p['proposed_action'] = response['action']
        snap = snapshot(PublicIncident.from_dict(p),runtime,response['raw_response']); save_snapshot(journal.root,snap)
        replay = no_edit_replay(victim,snap,journal,budget=len(plan['tasks']))
        record = {'task_fingerprint':task['task_fingerprint'], 'dependence_group':task['dependence_group'],
                  'cohort':task['cohort'], 'instruction':task['instruction'], 'snapshot_hash':snap['hash'],
                  'status':replay['status'], 'replay':replay, 'measurements':action_metrics(p,response['action'])}
        records.append(record); immutable_json(journal.root/'cases'/(p['case_id']+'.json'),record)
    return {'protocol':VERSION,'status':'completed' if all(r['status']=='replay_valid' for r in records) else 'failed',
            'simulated':plan['simulated'],'scientific_confirmation':False,'checkpoint_alias':row['checkpoint_alias'],
            'cohorts':summarize(records),'records':records,'limitations':LIMITATIONS,'clean_control':plan['clean_control']}


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--config',required=True); prep.add_argument('--output',required=True)
    prep.add_argument('--per-cohort',type=int,default=4)
    audit = sub.add_parser('audit-opportunities'); audit.add_argument('--run-root',required=True); audit.add_argument('--output',required=True)
    run = sub.add_parser('run'); run.add_argument('--plan',required=True); run.add_argument('--row',type=int,required=True)
    run.add_argument('--output',required=True); run.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    try:
        destination = fresh(args.output)
        if args.command == 'prepare':
            plan = prepare(read_json(resolve(args.config)),args.per_cohort); plan['hash']=digest(plan)
            destination.mkdir(parents=True,exist_ok=False); immutable_json(destination/'plan.json',plan)
            print(json.dumps({k:v for k,v in plan.items() if k not in ('tasks','source')},indent=2)); return 0
        if args.command == 'audit-opportunities':
            snaps = []
            for row in sorted(Path(args.run_root).glob('real/row-*')):
                snaps.extend(load_snapshots(row))
            if not snaps: raise Invalid('no real snapshot rows')
            result = opportunity_audit(snaps); immutable_json(destination,result)
            print(json.dumps(result['counts'],indent=2)); return 0
        plan = read_json(args.plan)
        if plan['hash'] != digest({k:v for k,v in plan.items() if k!='hash'}) or plan['source'] != source() or plan['protocol'] != VERSION:
            raise Invalid('plan/source mismatch; prepare a fresh characterization run')
        if not 0 <= args.row < len(plan['rows']): raise Invalid('invalid row')
        row = plan['rows'][args.row]; entry = checkpoint_entry(plan,row)
        if not entry['enabled'] or entry['identity'] != row['checkpoint_identity'] or digest(entry['weights']) != entry['identity']:
            raise Invalid('checkpoint/plan mismatch')
        if not (resolve(entry['path'])/'config.json').is_file(): raise Invalid('local checkpoint missing')
        if args.dry_run:
            print(json.dumps({'status':'ready','tasks':len(plan['tasks']), 'max_victim_calls':plan['max_victim_calls_per_row'],
                              'checkpoint_alias':row['checkpoint_alias'],'model_calls':0},indent=2)); return 0
        if not os.environ.get('SLURM_JOB_ID'): raise Invalid('real generation requires Slurm')
        if plan['simulated']: raise Invalid('real worker rejects simulated plans')
        destination.mkdir(parents=True,exist_ok=False)
        immutable_json(destination/'manifest.json',{'plan':plan,'row':args.row,'slurm_job_id':os.environ['SLURM_JOB_ID']})
        journal = Journal(destination)
        result = {'protocol':VERSION,'status':'failed','simulated':False,'scientific_confirmation':False}
        try:
            def interrupted(signum, frame): raise InterruptedError('Slurm termination')
            signal.signal(signal.SIGTERM,interrupted)
            verify_weights(entry)
            from seek.victim import LegacyVictim
            victim = LegacyVictim(dict(entry,path=str(resolve(entry['path']))),plan['generation'])
            immutable_json(destination/'runtime.json',{'weights_verified':True,'backend':victim.backend_identity,
                            'gpu':victim.torch.cuda.get_device_name(0),'visible_device':0})
            result = probe(plan,args.row,victim,journal)
        except Exception as exc:
            result.update(error_type=type(exc).__name__,reason=str(exc) if isinstance(exc,Invalid) else 'Inspect logs/events')
        result['accounting']=call_metrics(journal.records); immutable_json(destination/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
        return 0 if result['status']=='completed' else 2
    except (Invalid,OSError,ValueError,KeyError) as exc:
        parser.error(str(exc))

if __name__=='__main__': raise SystemExit(main())
