#!/usr/bin/env python3
"""Synthetic scalar-loss arithmetic/autograd checks; no model or optimizer."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from propose_v2_exact_passes import schedule,verify
from prepare_lexical_v2 import ROOT
from seek.discovery_phase.common import seal
from seek.three_way.lexical_pair import need
from verify_csham_review_packet_v3 import sha


def token_values(record):
    # Unequal valid-token counts expose accidental global-token weighting.
    return [F((record%7)+j+1,3) for j in range(1+record%5)]


def example_gradient(record):
    targets=token_values(record)
    return sum(2*(F(1,2)-v) for v in targets)/len(targets)


def check_window(window,world):
    steps=window['microsteps'];valid=[i for row in steps for i in row if i is not None]
    need(bool(valid) and len(valid)==window['valid_records'],'window_count')
    need(all(len(row)==world for row in steps),'rank_width')
    reference=sum(example_gradient(i) for i in valid)/len(valid)
    rank_sums=[F(0) for _ in range(world)];trace=[]
    for step,row in enumerate(steps):
        for rank,i in enumerate(row):
            rank_sums[rank]+=(example_gradient(i) if i is not None else F(0))*world/len(valid)
            trace.append((step,rank,'zero' if i is None else 'real'))
    reduced=sum(rank_sums)/world
    need(reference==reduced,'gradient_scaling_mismatch')
    need(len(trace)==len(steps)*world,'participation_slots')
    return dict(reference=reference,reduced=reduced,trace=trace)


def autograd_window(window,world):
    import torch
    valid=[i for row in window['microsteps'] for i in row if i is not None]
    reference=torch.tensor(0.5,dtype=torch.float64,requires_grad=True,device='cpu')
    def loss(p,i):
        targets=torch.tensor([float(v) for v in token_values(i)],dtype=torch.float64,device='cpu')
        return ((p-targets)**2).mean()
    sum(loss(reference,i) for i in valid).div(len(valid)).backward()
    params=[torch.tensor(0.5,dtype=torch.float64,requires_grad=True,device='cpu') for _ in range(world)]
    for row in window['microsteps']:
        for rank,i in enumerate(row):
            # Connected scalar zero: never mean over an empty/all-ignored batch.
            contribution=loss(params[rank],i) if i is not None else params[rank]*0
            (contribution*world/len(valid)).backward()
    need(all(p.grad is not None and torch.isfinite(p.grad).item() for p in params),'empty_rank_gradient')
    reduced=sum(p.grad for p in params)/world
    need(torch.allclose(reduced,reference.grad,rtol=1e-12,atol=1e-12),'autograd_reference_mismatch')
    return abs(float(reduced-reference.grad))


def run(backend):
    need(backend in ('rational','torch'),'backend')
    rows=[];max_error=0.0
    for world in (1,2,4,8):
        windows=schedule(1902,world);verify(windows,1902,world)
        zero_slots=0
        for window in windows:
            result=check_window(window,world)
            zero_slots+=sum(t[2]=='zero' for t in result['trace'])
            if backend=='torch':max_error=max(max_error,autograd_window(window,world))
        rows.append(dict(world_size_assumption=world,windows_checked=len(windows),empty_slots_checked=zero_slots))
    # Each rank except one is empty in this separate stress fixture.
    stress=schedule(1,8,epochs=1)[0];check_window(stress,8)
    if backend=='torch':max_error=max(max_error,autograd_window(stress,8))
    versions={}
    if backend=='torch':
        import torch
        versions['torch']=torch.__version__
    names=['scripts/seek/check_v2_loss_scaling.py','scripts/seek/propose_v2_exact_passes.py']
    return seal(dict(schema='seek-v2-synthetic-loss-check-v1',status='synthetic_loss_checks_passed',backend=backend,
        cases=rows,empty_rank_stress_fixture_passed=True,maximum_absolute_autograd_error=max_error if backend=='torch' else None,
        objective='mean of per-example token-mean squared losses at scalar parameter 0.5',
        scale='world_size / actual_valid_records_in_window; arithmetic mean of rank gradients; no extra accumulation division',
        source_files={n:sha((ROOT/n).read_bytes()) for n in names},versions=versions,
        actual_collectives_executed=False,ddp_reducer_verified=False,trainer_instantiated=False,
        scope='synthetic scalar arithmetic; torch mode additionally uses scalar autograd, not a model forward/backward',
        limitations=['Rank reduction is arithmetic, not distributed execution',
          'A connected scalar zero does not establish all-parameter DDP hook correctness for a real model',
          'FSDP, AMP, gradient clipping, optimizer and scheduler behavior untested',
          'No training entry installed; Accelerate automatic accumulation scaling must not be applied again'],
        model_calls=0,optimizer_steps=0,training_authorized=False,execution_authorized=False,installation_authorized=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('rational','torch'),default='rational');p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.backend)
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
