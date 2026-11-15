#!/usr/bin/env python3
"""Single-process CPU sampler replay and explicit Trainer-control-flow emulation."""
import argparse
from collections import Counter
import inspect
import json
import math
from pathlib import Path
from prepare_lexical_v2 import ROOT
from seek.discovery_phase.common import seal
from seek.three_way.lexical_pair import need
from verify_csham_review_packet_v3 import sha

SOURCE_HASH='8debfb20c6ae58eff3a466cfd67244a12cc8b8d3de00316a6c1d54b0310ee6ca'


def control_trace(length,epochs=2,accumulation=8):
    need(all(type(v) is int and v>0 for v in (length,epochs,accumulation)),'positive_control_inputs')
    maximum=math.ceil(epochs*max(length//accumulation,1))
    total=updates=0;window=[];events=[];consumed=[]
    for epoch in range(epochs):
        count=0
        for step in range(length):
            total+=1;count+=1;window.append([epoch,step])
            if total%accumulation==0 or (length<=accumulation and step+1==length):
                updates+=1;events.append(dict(update=updates,microbatches=window));window=[]
            if updates>=maximum:break
        consumed.append(count)
        if updates>=maximum:break
    return dict(planned_updates=maximum,step_events=events,consumed_batches_per_rank_by_epoch=consumed,
        pending_microbatches=window,cross_epoch_windows=sum(len({x[0] for x in e['microbatches']})>1 for e in events))


def run():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from torch.utils.data import RandomSampler,BatchSampler
    import accelerate
    from accelerate.data_loader import BatchSamplerShard
    source=Path(inspect.getsourcefile(BatchSamplerShard));need(sha(source.read_bytes())==SOURCE_HASH,'installed_sharder_source_changed')
    cases=[]
    for world in (1,2,4,8):
        trace=control_trace(math.ceil(1902/world));counts=Counter();epoch_rows=[]
        for epoch,consumed in enumerate(trace['consumed_batches_per_rank_by_epoch']):
            ranks=[]
            for rank in range(world):
                # Explicit diagnostic order, not an assertion of Trainer RNG state.
                generator=torch.Generator(device='cpu').manual_seed(271028+epoch)
                sampler=RandomSampler(range(1902),generator=generator)
                batch=BatchSampler(sampler,batch_size=1,drop_last=False)
                shard=BatchSamplerShard(batch,num_processes=world,process_index=rank,split_batches=False,even_batches=True)
                batches=list(shard);need(len(batches)==math.ceil(1902/world),'shard_length')
                ranks.append(batches)
            all_ids=[i for r in ranks for b in r for i in b]
            seen=[i for r in ranks for b in r[:consumed] for i in b];counts.update(seen)
            epoch_rows.append(dict(epoch=epoch,full_shard_examples=len(all_ids),full_shard_unique=len(set(all_ids)),
                full_shard_padding_repeats=len(all_ids)-len(set(all_ids)),consumed_examples=len(seen),
                consumed_unique=len(set(seen)),omitted_distinct_ids=sorted(set(range(1902))-set(seen)),
                consumed_order_hash=sha(json.dumps(seen).encode())))
        cases.append(dict(world_size_assumption=world,control=trace,epochs=epoch_rows,
            exposure_histogram={str(n):sum(counts[i]==n for i in range(1902)) for n in sorted({counts[i] for i in range(1902)})}))
    return seal(dict(schema='seek-v2-sampler-control-diagnostic-v1',status='cpu_sampler_diagnostic_complete',
        records=1902,epochs=2,microbatch=1,accumulation=8,cases=cases,
        package_versions=dict(torch=torch.__version__,accelerate=accelerate.__version__),
        sharder_source_sha256=sha(source.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
        sampler='actual RandomSampler + BatchSampler + BatchSamplerShard; simulated ranks in one CPU process',
        order_policy='diagnostic seed 271028+epoch; not actual Trainer/Accelerator RNG replay',
        control_policy='explicit emulation of reviewed floor planning, cross-epoch counter and ordinary max-step stop',
        limitations=['No real Trainer/callback execution','No distributed synchronization or gradients','No optimizer or model','No guarantee of actual training sample order'],
        recipe_changed=False,topology_selected=False,model_calls=0,training_authorized=False,execution_authorized=False,installation_authorized=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    r=run()
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(dict(status=r['status'],hash=r['hash'],cases=[dict(world_size=x['world_size_assumption'],planned_updates=x['control']['planned_updates'],cross_epoch_windows=x['control']['cross_epoch_windows'],exposure_histogram=x['exposure_histogram']) for x in r['cases']],model_calls=0,training_authorized=False),indent=2))
