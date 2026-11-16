#!/usr/bin/env python3
"""Synthetic multi-parameter DDP + causal-CE fixture; no agent weights or optimizer."""
import argparse
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from exact_pass_runtime import BoundWindows, causal_example_means, backward_ddp_window
from propose_v2_exact_passes import schedule
from check_torch_environment_identity import assess, collect
from prepare_lexical_v2 import ROOT
from seek.discovery_phase.common import seal
from seek.three_way.lexical_pair import need
from verify_csham_review_packet_v3 import sha


def fixture(torch):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float64).reshape(3, 4) / 20)
            self.bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float64) / 10)
        def forward(self, x):
            return x @ self.weight + self.bias
    return Toy()


def sample(torch, index):
    x = torch.arange(15, dtype=torch.float64).reshape(1, 5, 3) / 17 + index / 31
    labels = torch.tensor([[-100, 1, 2, 3, 0]], dtype=torch.long)
    labels[:, 1:1 + index % 3] = -100
    return x, labels


def independent_loss(torch, model, index):
    x, labels = sample(torch, index)
    logp = model(x).log_softmax(-1)
    terms = [-logp[0, t-1, int(labels[0, t])] for t in range(1, 5) if labels[0, t] != -100]
    return torch.stack(terms).mean()


def worker(rank, world, rendezvous, output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=world, timeout=timedelta(seconds=120))
    try:
        model = fixture(torch); ddp = DDP(model)
        # Full accumulation, partial windows, and an empty-rank-only stress window.
        windows = schedule(world * 8 + 1, world, epochs=1) + schedule(1, world, epochs=1)
        errors = []; forwards = 0; empty = 0
        for window in windows:
            ddp.zero_grad(set_to_none=True)
            reference = fixture(torch)
            valid = [i for s in window['microsteps'] for i in s if i is not None]
            (sum(independent_loss(torch, reference, i) for i in valid) / len(valid)).backward()
            slots = [s[rank] for s in window['microsteps']]
            def forward_loss(wrapped, index):
                nonlocal forwards, empty
                forwards += 1
                x, labels = sample(torch, index if index is not None else 0)
                logits = wrapped(x)
                if index is None:
                    empty += 1
                    # Every toy parameter participates through the actual DDP forward.
                    return logits.sum() * 0
                return causal_example_means(logits, labels)[0]
            backward_ddp_window(ddp, slots, len(valid), world, forward_loss)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                need(actual.grad is not None, 'missing_parameter_hook')
                error = (actual.grad - expected.grad).abs().max().item()
                need(torch.allclose(actual.grad, expected.grad, atol=1e-12, rtol=1e-12), 'ddp_gradient')
                errors.append(error)
        report = dict(rank=rank, world=world, windows=len(windows), forwards=forwards,
                      empty_slots=empty, maximum_absolute_error=max(errors), parameter_tensors=2)
        (Path(output) / f'rank-{rank}.json').write_text(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


def ce_checks():
    import torch
    logits = torch.arange(40, dtype=torch.float64).reshape(2, 5, 4).div(23).requires_grad_()
    labels = torch.tensor([[-100, 1, 2, 3, 0], [-100, -100, -100, -100, 2]])
    means = causal_example_means(logits, labels)
    manual = []
    for b in range(2):
        terms = [-logits[b, t-1].log_softmax(-1)[labels[b, t]] for t in range(1, 5) if labels[b, t] != -100]
        manual.append(torch.stack(terms).mean())
    reference = torch.stack(manual).mean()
    gradient = torch.autograd.grad(means.mean(), logits, retain_graph=True)[0]
    expected = torch.autograd.grad(reference, logits)[0]
    need(torch.allclose(gradient, expected, atol=1e-12, rtol=1e-12), 'ce_gradient')
    need(not torch.allclose(means.mean(), (means[0]*4+means[1])/5), 'weighting_fixture_not_discriminative')
    need(bool((gradient[:, -1] == 0).all()), 'last_logit_shift')
    try:
        causal_example_means(logits, torch.full_like(labels, -100))
    except Exception as exc:
        need(str(exc) == 'no_shifted_supervision', 'wrong_empty_rejection')
    else:
        raise ValueError('all_masked_accepted')
    return dict(variable_target_lengths=True, shifted_gradient_verified=True, all_masked_rejected=True)


def bind_candidate(path):
    from verify_csham_review_packet_v3 import archive
    from check_v2_saved_trainer_inputs import ZIP_SHA, PLAN
    from seek.discovery_phase.common import checked
    raw = Path(path).read_bytes(); need(sha(raw) == ZIP_SHA, 'native_zip_identity')
    outer = archive(raw); candidate = archive(outer.read('candidate.review.private.zip'))
    plan = checked(json.loads(candidate.read('plan.private.json')))
    need(plan['hash'] == PLAN, 'candidate_identity')
    bindings = {}; paired_ids = None
    for role in ('C', 'LQ'):
        records = json.loads(candidate.read(role+'.records.private.json'))
        annotations = json.loads(candidate.read(role+'.annotations.private.json'))
        need(len(records) == 1902, 'candidate_count')
        bindings[role] = []
        for world in (1, 2, 4, 8):
            bound = BoundWindows(records, annotations, plan['corpus_hashes'][role],
                                 plan['annotation_hashes'][role], world)
            if paired_ids is None: paired_ids = bound.ids
            need(bound.ids == paired_ids, 'paired_order')
            for rank in range(world):
                for window in bound.rank_batches(rank):
                    for slot in window['slots']:
                        if slot is not None: bound.fetch(records, slot)
            bindings[role].append(dict(binding=bound.binding, summary=bound.summary))
    return dict(native_zip_sha256=ZIP_SHA, candidate_hash=PLAN, roles=bindings)


def run(output, native_zip):
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', GLOO_SOCKET_IFNAME='lo', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import torch
    import torch.multiprocessing as mp
    need(torch.__version__ == '2.6.0+cpu', 'isolated_cpu_required')
    identity = collect(); need(not assess(identity)['failures'], 'environment_identity')
    out = Path(output); out.mkdir(parents=True, exist_ok=False, mode=0o700)
    candidate = bind_candidate(native_zip)
    ce = ce_checks(); cases = []
    for world in (2, 4):
        folder = out / f'world-{world}'; folder.mkdir()
        with tempfile.TemporaryDirectory(prefix='seek-ddp-') as temp:
            mp.spawn(worker, args=(world, str(Path(temp)/'rendezvous'), str(folder)), nprocs=world, join=True)
        ranks = [json.loads((folder/f'rank-{r}.json').read_text()) for r in range(world)]
        need(len({r['forwards'] for r in ranks}) == 1, 'participation')
        cases.append(dict(world=world, ranks=ranks))
    names = ['scripts/seek/exact_pass_runtime.py', 'scripts/seek/check_exact_pass_ddp_cpu.py',
             'scripts/seek/propose_v2_exact_passes.py', 'scripts/seek/check_torch_environment_identity.py']
    report = seal(dict(schema='seek-exact-pass-ddp-fixture-v1', status='synthetic_ddp_and_causal_ce_passed',
        identity=identity, candidate_binding=candidate, ce_checks=ce, cases=cases, source_files={n:sha((ROOT/n).read_bytes()) for n in names},
        scope='CPU toy linear module, two parameter tensors, actual DDP reducer and no_sync; no agent model',
        real_model_empty_path_verified=False, real_model_loss_verified=False,
        synthetic_module_forwards=sum(r['forwards'] for c in cases for r in c['ranks']),
        model_calls=0, model_calls_scope='agent model calls; toy forwards counted separately', optimizer_steps=0,
        training_authorized=False, execution_authorized=False, installation_authorized=False))
    (out/'report.private.json').write_text(json.dumps(report, indent=2)+'\n')
    return dict(status=report['status'], hash=report['hash'], ce_checks=ce,
        maximum_absolute_error=max(r['maximum_absolute_error'] for c in cases for r in c['ranks']),
        synthetic_module_forwards=report['synthetic_module_forwards'], optimizer_steps=0, training_authorized=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); p.add_argument('--native-zip', required=True)
    a=p.parse_args(); print(json.dumps(run(a.output, a.native_zip), indent=2))
