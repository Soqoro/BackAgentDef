#!/usr/bin/env python3
"""Single-host Gloo scalar-gradient diagnostic; no model, optimizer or Trainer."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from check_v2_loss_scaling import token_values,example_gradient
from propose_v2_exact_passes import schedule,verify
from prepare_lexical_v2 import ROOT
from seek.discovery_phase.common import seal
from seek.three_way.lexical_pair import need
from verify_csham_review_packet_v3 import sha
from check_torch_environment_identity import assess,collect


def rank_window(window,rank,world):
    need(type(rank) is int and 0<=rank<world,'rank')
    need(bool(window['microsteps']) and all(len(s)==world for s in window['microsteps']),'rank_width')
    valid=[i for s in window['microsteps'] for i in s if i is not None]
    need(len(valid)==window['valid_records'] and bool(valid),'valid_count')
    return [s[rank] for s in window['microsteps']],valid


def worker(rank,world,rendezvous,out):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=Path(rendezvous).as_uri(),rank=rank,world_size=world,timeout=timedelta(seconds=120))
    try:
        windows=schedule(1902,world);verified=verify(windows,1902,world)
        stress=schedule(1,world,epochs=1)[0]
        exposure=torch.zeros(1902,dtype=torch.int64);errors=[];zero_slots=0;backwards=0
        for index,window in enumerate(windows+[stress]):
            local,valid=rank_window(window,rank,world)
            # This count collective checks agreement on the real-example denominator.
            count=torch.tensor(sum(i is not None for i in local),dtype=torch.int64)
            dist.all_reduce(count,op=dist.ReduceOp.SUM)
            need(count.item()==len(valid),'collective_count_mismatch')
            parameter=torch.tensor(0.5,dtype=torch.float64,requires_grad=True)
            for i in local:
                if i is None:
                    loss=parameter*0;zero_slots+=1
                else:
                    targets=torch.tensor([float(v) for v in token_values(i)],dtype=torch.float64)
                    loss=((parameter-targets)**2).mean()
                    if index<len(windows):exposure[i]+=1
                (loss*world/int(count.item())).backward();backwards+=1
            need(parameter.grad is not None and torch.isfinite(parameter.grad).item(),'finite_gradient')
            reduced=parameter.grad.detach().clone()
            dist.all_reduce(reduced,op=dist.ReduceOp.SUM);reduced/=world
            expected=float(sum(example_gradient(i) for i in valid)/len(valid))
            error=abs(reduced.item()-expected);need(error<=1e-12*(1+abs(expected)),'collective_gradient_mismatch')
            errors.append(error)
        dist.all_reduce(exposure,op=dist.ReduceOp.SUM)
        need(torch.equal(exposure,torch.full_like(exposure,2)),'actual_rank_exposure_mismatch')
        report=dict(rank=rank,world_size=world,windows=len(windows),stress_windows=1,
            backwards=backwards,empty_slots=zero_slots,collectives=2*(len(windows)+1)+1,
            maximum_absolute_error=max(errors),all_records_exposed_exactly_twice=True,
            schedule_hash=sha(json.dumps(windows,sort_keys=True).encode()),schedule_summary=verified)
        with (Path(out)/f'rank-{rank}.json').open('x') as f:json.dump(report,f,indent=2)
    finally:dist.destroy_process_group()


def run(output):
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GLOO_SOCKET_IFNAME='lo',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import torch
    import torch.multiprocessing as mp
    need(torch.__version__=='2.6.0+cpu','expected_isolated_cpu_torch')
    identity=collect();need(not assess(identity)['failures'],'inconsistent_environment')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700)
    cases=[]
    for world in (2,4):
        case=out/f'world-{world}';case.mkdir()
        # Local file rendezvous and loopback collectives; no external service or launcher.
        with tempfile.TemporaryDirectory(prefix='seek-gloo-') as temp:
            mp.spawn(worker,args=(world,str(Path(temp)/'rendezvous'),str(case)),nprocs=world,join=True)
        ranks=[json.loads((case/f'rank-{rank}.json').read_text()) for rank in range(world)]
        need(len({r['schedule_hash'] for r in ranks})==1,'rank_schedule_disagreement')
        need(len({r['backwards'] for r in ranks})==1,'rank_participation_disagreement')
        cases.append(dict(world_size=world,windows=ranks[0]['windows'],stress_windows=1,
            maximum_absolute_error=max(r['maximum_absolute_error'] for r in ranks),
            empty_slots=sum(r['empty_slots'] for r in ranks),
            collective_calls_per_rank=ranks[0]['collectives'],all_records_exposed_exactly_twice=True,
            ranks=ranks))
    names=['scripts/seek/check_v2_cpu_collectives.py','scripts/seek/check_v2_loss_scaling.py','scripts/seek/propose_v2_exact_passes.py','scripts/seek/check_torch_environment_identity.py']
    report=seal(dict(schema='seek-v2-cpu-collective-check-v1',status='synthetic_gloo_collectives_passed',
        identity=identity,cases=cases,backend='gloo',actual_collectives_executed=True,
        source_files={n:sha((ROOT/n).read_bytes()) for n in names},
        scope='Single-host scalar autograd with explicit all_reduce; synthetic losses and record indices only',
        limitations=['No DDP reducer or real-model parameter hooks tested','No FSDP, AMP, clipping, optimizer or scheduler tested','No production training entry installed','Synthetic order is not a trained-model result'],
        ddp_reducer_verified=False,model_calls=0,optimizer_steps=0,training_authorized=False,installation_authorized=False))
    (out/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k not in ('identity','source_files','cases')} | {'cases':[{k:v for k,v in c.items() if k!='ranks'} for c in cases]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.output),indent=2))
