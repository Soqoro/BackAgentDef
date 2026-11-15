#!/usr/bin/env python3
"""CPU-only schedule proposal. Not a sampler, loader or training entry."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
from prepare_lexical_v2 import ROOT
from verify_csham_review_packet_v3 import archive,sha
from seek.discovery_phase.common import checked,seal
from seek.schemas import digest
from seek.three_way.lexical_pair import need
from check_v2_saved_trainer_inputs import ZIP_SHA,PLAN


def schedule(count,world,accumulation=8,epochs=2,seed=271028):
    need(all(type(x) is int and x>0 for x in (count,world,accumulation,epochs)),'positive_schedule_parameters')
    need(type(seed) is int and seed>=0,'seed')
    windows=[]
    for epoch in range(epochs):
        # Explicit versioned permutation; independent of library RNG implementations.
        order=sorted(range(count),key=lambda i:(sha(f'exact-pass-v1:{seed}:{epoch}:{i}'.encode()),i))
        for start in range(0,count,world*accumulation):
            members=order[start:start+world*accumulation]
            padded=members+[None]*((-len(members))%world)
            windows.append(dict(epoch=epoch,index=len(windows),valid_records=len(members),
                microsteps=[padded[j:j+world] for j in range(0,len(padded),world)],
                partial=len(members)<world*accumulation))
    return windows


def verify(windows,count,world,epochs=2):
    seen=Counter();padding=0;steps=0
    for window in windows:
        valid=[i for step in window['microsteps'] for i in step if i is not None]
        need(len(valid)==window['valid_records'] and len(set(valid))==len(valid),'window_membership')
        need(all(len(step)==world for step in window['microsteps']),'rank_width')
        need(all(type(i) is int and 0<=i<count for i in valid),'record_index')
        seen.update((window['epoch'],i) for i in valid)
        padding+=sum(i is None for step in window['microsteps'] for i in step)
        steps+=len(window['microsteps'])
        # Symbolic DDP mean: each valid example mean-loss has coefficient W/N
        # before rank averaging, giving exactly 1/N after. No extra /accumulation.
        coefficient=Fraction(world,len(valid))/world
        need(coefficient==Fraction(1,len(valid)),'normalization')
    need(seen==Counter({(e,i):1 for e in range(epochs) for i in range(count)}),'exact_epoch_exposure')
    return dict(updates=len(windows),microsteps=steps,valid_presentations=sum(seen.values()),
        empty_rank_slots=padding,records_with_exactly_two_exposures=count if epochs==2 else None,
        cross_epoch_windows=0,partial_windows=sum(w['partial'] for w in windows))


def prepare(path,output):
    raw=Path(path).read_bytes();need(sha(raw)==ZIP_SHA,'native_zip_identity')
    outer=archive(raw);c=archive(outer.read('candidate.review.private.zip'))
    plan=checked(json.loads(c.read('plan.private.json')));need(plan['hash']==PLAN,'candidate_identity')
    corpora={r:json.loads(c.read(r+'.records.private.json')) for r in ('C','LQ')}
    for role,records in corpora.items():need(digest(records)==plan['corpus_hashes'][role],'corpus_hash')
    ids=[r['id'] for r in corpora['C']]
    need(ids==[r['id'] for r in corpora['LQ']] and len(set(ids))==len(ids)==1902,'paired_ids')
    cases=[]
    for world in (1,2,4,8):
        windows=schedule(len(ids),world)
        cases.append(dict(world_size_assumption=world,summary=verify(windows,len(ids),world),windows=windows))
    report=seal(dict(schema='seek-v2-exact-pass-proposal-v1',status='cpu_schedule_checked_not_training_implementation',
        candidate_hash=PLAN,input_zip_sha256=sha(raw),paired_record_ids_hash=digest(ids),
        recipe_change_proposed=True,recipe_approved=False,topology_selected=False,
        permutation='SHA256(exact-pass-v1:seed:epoch:record_index), tie-break index',seed=271028,
        microbatch_per_rank=1,accumulation_max_microsteps=8,epochs=2,cases=cases,
        loss_proposal='mean of per-example valid-token mean losses within each update window',
        loss_rationale='keeps equal example weighting of microbatch-one means; explicitly normalizes partial windows',
        gradient_contract='under DDP rank averaging: local sum of example mean-losses * world_size / valid_records_in_window; no additional accumulation division',
        padding_contract='None is a schedule placeholder only, never a corpus record or an all-masked standard model batch',
        blockers=['No executable training loop installed',
            'Empty-rank slots require reviewed differentiable zero contribution and identical collective participation',
            'Actual gradient synchronization and scaling, including final partial windows, require separate verification',
            'Default Accelerate backward division must not be composed blindly with proposed scaling',
            'FSDP/DeepSpeed/other reduction semantics not established by the DDP algebra',
            'Initialization provenance, optimizer, hardware resources and final corpus/execution decisions unresolved'],
        checkpoint_selection_proposed='after completion of the final scheduled update of pass two',
        source_files={'scripts/seek/propose_v2_exact_passes.py':sha(Path(__file__).read_bytes())},
        transfer_attempts=132,optional_seen_context_enabled=False,optional_seen_context_attempts=32,
        model_calls=0,training_authorized=False,execution_authorized=False,installation_authorized=False))
    out=Path(output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/'proposal.private.json').write_text(json.dumps(report,indent=2)+'\n')
    summary=dict(status=report['status'],hash=report['hash'],cases=[dict(world_size=x['world_size_assumption'],**x['summary']) for x in cases],model_calls=0,training_authorized=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-zip',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.native_zip,a.output),indent=2))
