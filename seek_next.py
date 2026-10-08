#!/usr/bin/env python3
"""Prospective Seek experiments; CPU commands never load a model."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest
from seek.storage import read_json,immutable_json
from seek.semantic_registry import Registry
from seek.next_phase import plans,design,engine,audit,simulation,investigation


def parser():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 a=s.add_parser('audit');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--archive',action='store_true')
 a=s.add_parser('package');a.add_argument('--input',required=True);a.add_argument('--output',required=True)
 a=s.add_parser('profile');a.add_argument('--query',required=True);a.add_argument('--observation',required=True);a.add_argument('--reference',required=True);a.add_argument('--output',required=True)
 a=s.add_parser('compile');a.add_argument('--profile',required=True);a.add_argument('--kind',choices=['observation','query'],required=True);a.add_argument('--phase',choices=['pilot','confirmation'],default='pilot');a.add_argument('--mode',choices=['pilot','paper_anytime_v1','finite_support_census_v1'],default='pilot');a.add_argument('--request');a.add_argument('--reference',choices=['reference']);a.add_argument('--parent',type=int);a.add_argument('--output',required=True)
 for cmd in ('preview','resources','review-template'):
  a=s.add_parser(cmd);a.add_argument('--plan',required=True);a.add_argument('--study');a.add_argument('--output')
 a=s.add_parser('freeze');a.add_argument('--plan',required=True);a.add_argument('--review',required=True);a.add_argument('--study',required=True)
 for cmd in ('run-model','join','status'):
  a=s.add_parser(cmd);a.add_argument('--study',required=True);a.add_argument('--j',type=int,required=True)
  if cmd=='run-model':a.add_argument('--model',choices=['query','observation','reference']);a.add_argument('--model-index',type=int);a.add_argument('--allow-confirmation',action='store_true');a.add_argument('--dry-run',action='store_true')
  if cmd=='status':a.add_argument('--require',required=True)
 a=s.add_parser('export');a.add_argument('--study',required=True);a.add_argument('--output',required=True)
 a=s.add_parser('import-exposure');a.add_argument('--study',required=True);a.add_argument('--manifest',required=True)
 a=s.add_parser('simulate');a.add_argument('--replications',type=int,default=2);a.add_argument('--seed',type=int,default=86103);a.add_argument('--cap',type=int,default=1024);a.add_argument('--large',action='store_true');a.add_argument('--output',required=True)
 a=s.add_parser('bench-compile');a.add_argument('--incidents',required=True);a.add_argument('--schedule',required=True);a.add_argument('--budget',type=int,default=32);a.add_argument('--output',required=True)
 a=s.add_parser('bench-incidents');a.add_argument('--rows',nargs=2,required=True);a.add_argument('--output',required=True)
 a=s.add_parser('bench-run');a.add_argument('--config',required=True);a.add_argument('--index',type=int,required=True);a.add_argument('--variant',choices=investigation.VARIANTS,required=True);a.add_argument('--oracle',required=True);a.add_argument('--qwen-config',required=True);a.add_argument('--output',required=True)
 a=s.add_parser('bench-evaluate');a.add_argument('--candidates',nargs='+',required=True);a.add_argument('--assignments',required=True);a.add_argument('--study',required=True);a.add_argument('--output',required=True)
 a=s.add_parser('bench-oracle');a.add_argument('--study',required=True);a.add_argument('--config',required=True);a.add_argument('--j',nargs='+',type=int,required=True);a.add_argument('--output',required=True)
 a=s.add_parser('eval-prepare');a.add_argument('--config',required=True);a.add_argument('--candidates',nargs='+',required=True);a.add_argument('--profile',required=True);a.add_argument('--output',required=True)
 a=s.add_parser('eval-freeze');a.add_argument('--study',required=True);a.add_argument('--bank',required=True);a.add_argument('--review',required=True)
 for cmd in ('eval-model','eval-join'):
  a=s.add_parser(cmd);a.add_argument('--study',required=True);a.add_argument('--j',type=int,required=True)
  if cmd=='eval-model':a.add_argument('--model',choices=['query','observation','reference'],required=True)
 return p


def gpu_guard():
 if not os.environ.get('SLURM_JOB_ID'): raise Invalid('real inference requires a user-submitted Slurm allocation')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def main(argv=None):
 args=parser().parse_args(argv);cmd=args.command;result=None
 if cmd=='audit':
  if args.archive:
   with tempfile.TemporaryDirectory(prefix='seek-audit-') as tmp: audit.unpack(args.input,tmp);result=audit.report(tmp,args.output)
  else:result=audit.report(args.input,args.output)
 elif cmd=='package':result=audit.package(args.input,args.output)
 elif cmd=='profile':
  models={};generation=None
  for name in ('query','observation','reference'):
   manifest=read_json(getattr(args,name));entry=manifest['provenance']['checkpoint'];entry=dict(entry,path=manifest['provenance']['path'])
   if digest(entry['weights'])!=entry['identity']:raise Invalid('interface checkpoint identity changed')
   if generation is not None and generation!=manifest['generation']:raise Invalid('interface generations differ')
   generation=manifest['generation'];models[name]=entry
  result=dict(models=models,generation=generation,source_manifests={k:digest(read_json(getattr(args,k))) for k in models});immutable_json(args.output,result)
 elif cmd=='compile':
  profile=read_json(args.profile);result=plans.compile_plan(args.kind,args.phase,args.mode,profile['models'],profile['generation'],read_json(args.request) if args.request else None,args.reference,args.parent);immutable_json(args.output,result)
 elif cmd in ('preview','resources','review-template'):
  plan=plans.validate(read_json(args.plan))
  if cmd=='preview':result=design.preview(plan)
  elif cmd=='resources':result=plans.resources(plan,Registry(args.study) if args.study else None)
  else:result=dict(plan_hash=plan['hash'],accepted=False,resource_approved=False,independent=True,reviewer='',reason='',pilot_result_hash=None)
  if args.output:immutable_json(args.output,result)
 elif cmd=='freeze':result=plans.allocate(Registry(args.study),read_json(args.plan),read_json(args.review))
 elif cmd in ('run-model','join','status'):
  reg=Registry(args.study);plan=plans.load(reg,args.j)
  if cmd=='run-model':
   if args.dry_run:result=plans.resources(plan,reg)
   else:
    gpu_guard()
    if plan['phase']=='confirmation' and not args.allow_confirmation:raise Invalid('separate reviewed confirmation requires explicit --allow-confirmation')
    from seek.victim import LegacyVictim
    from seek.provenance import verify_weights
    def loader(entry,g):verify_weights(entry);v=LegacyVictim(entry,g);v.simulated=False;return v
    model=args.model or ['query','observation','reference'][args.model_index if args.model_index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])]
    result=engine.run_model(reg,plan,model,loader)
  else:
   result=engine.join(reg,plan,write=cmd=='join')
   if cmd=='status' and result['status']!=args.require:print(json.dumps(result,indent=2));return 2
 elif cmd=='export':result=audit.report(args.study,args.output)
 elif cmd=='import-exposure':
  manifest=read_json(args.manifest);result=Registry(args.study).expose(manifest['fingerprints'],dict(source_manifest=digest(manifest),scope='historical exposure; verification gaps retained'))
 elif cmd=='simulate':result=simulation.run(args.replications,args.seed,args.cap,large=args.large);immutable_json(args.output,result)
 elif cmd=='bench-compile':result=investigation.compile_benchmark(read_json(args.incidents),read_json(args.schedule),args.budget);immutable_json(args.output,result)
 elif cmd=='bench-incidents':result=investigation.select_incidents(args.rows);immutable_json(args.output,result)
 elif cmd=='bench-run':
  gpu_guard();config=read_json(args.qwen_config);backend=investigation.Roles(config['agents'])
  try:result=investigation.investigate(read_json(args.config),args.index,args.variant,backend,read_json(args.oracle),args.output)
  finally:backend.close()
 elif cmd=='bench-evaluate':result=investigation.evaluate([read_json(p) for p in args.candidates],read_json(args.assignments),Registry(args.study));immutable_json(args.output,result)
 elif cmd=='bench-oracle':result=investigation.make_oracle(read_json(args.config),Registry(args.study),args.j);immutable_json(args.output,result)
 elif cmd=='eval-prepare':result=investigation.prepare_evaluation(read_json(args.config),[read_json(p) for p in args.candidates],read_json(args.profile));immutable_json(args.output,result)
 elif cmd=='eval-freeze':result=investigation.freeze_evaluation(Registry(args.study),read_json(args.bank),read_json(args.review))
 elif cmd in ('eval-model','eval-join'):
  reg=Registry(args.study);bank=investigation.load_evaluation(reg,args.j)
  if cmd=='eval-join':result=investigation.join_evaluation(reg,bank)
  else:
   gpu_guard()
   from seek.victim import LegacyVictim
   from seek.provenance import verify_weights
   def loader(entry,g):verify_weights(entry);v=LegacyVictim(entry,g);v.simulated=False;return v
   result=investigation.run_evaluation_model(reg,bank,args.model,loader)
 print(json.dumps(result,indent=2));return 0

if __name__=='__main__':
 try:sys.exit(main())
 except (Invalid,ValueError,KeyError,FileNotFoundError) as e:
  print(json.dumps(dict(status='prerequisite_missing_or_invalid',error=str(e),model_calls_not_inferred=True)),file=sys.stderr);sys.exit(2)
