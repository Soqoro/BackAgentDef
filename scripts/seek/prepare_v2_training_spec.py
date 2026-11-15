#!/usr/bin/env python3
"""Bind the reviewed v2 candidate to a non-executable training specification."""
import argparse
import json
from pathlib import Path
from prepare_lexical_v2 import ROOT
from verify_csham_review_packet_v3 import archive, sha
from bundle_lexical_v2_balanced_review import write_zip
from seek.discovery_phase.common import checked, seal
from seek.schemas import digest
from seek.three_way.lexical_pair import need
from seek.three_way.feasibility import sampler_accounting

CANDIDATE = 'c934d2677795188074ed3c4c466d110cdbdd1bdb0ef1bc9e0baf02b4651653f0'
REVIEWS = {
    'source': '83b775d0d0ca35365b0e6e273e4ad564a6415fd969ff0bc37351e58a7a99d446',
    'cpu': 'af2e29cf54ca05526f2d270d7b0df4f8520bc8ad7e03a68bf17ca9ff2087c374',
}


def proposed_recipe(parent):
    """Never substitute older template defaults for the bound paired proposal."""
    recipe = dict(parent['recipe'])
    need(recipe['optimizer'] is None and recipe['gpus'] is None, 'unexpected_parent_recipe')
    return dict(inherited=recipe, optimizer_selection=None, initialization_weight_manifest=None,
        trainable_parameter_manifest=None, topology=None, sampler_implementation=None,
        loss_normalization=None, precision_runtime=None, output_paths=None,
        expected_updates=sampler_accounting(1902, recipe['epochs'], recipe['batch_per_gpu'],
            recipe['gradient_accumulation'])['expected_updates'],
        checkpoint_selection=recipe['checkpoint_selection'], status='unresolved_not_launchable')


def prepare(candidate_dir, source_review, cpu_review, output):
    root = Path(candidate_dir)
    plan = checked(json.loads((root/'plan.private.json').read_bytes()))
    need(plan['hash'] == CANDIDATE, 'candidate_identity')
    review_bytes = {'source': Path(source_review).read_bytes(), 'cpu': Path(cpu_review).read_bytes()}
    for kind, raw in review_bytes.items():
        need(sha(raw) == REVIEWS[kind], 'review_bytes_changed:'+kind)
    candidate_zip = (root/'review.private.zip').read_bytes()
    z = archive(candidate_zip)
    need(json.loads(z.read('plan.private.json')) == plan, 'candidate_archive_plan')
    parent_raw = z.read('parent.review.private.zip')
    need(sha(parent_raw) == plan['parent_packet_sha256'], 'parent_archive')
    parent = checked(json.loads(archive(parent_raw).read('candidate/proposal.private.json')))
    need(parent['hash'] == plan['parent_proposal_hash'], 'parent_proposal')
    for role in ('C','LQ'):
        for suffix, key in (('records','corpus_hashes'),('annotations','annotation_hashes')):
            name = role+'.'+suffix+'.private.json'
            raw = (root/name).read_bytes()
            need(raw == z.read(name), 'candidate_archive_member:'+name)
            rows = json.loads(raw)
            need(len(rows) == 1902 and digest(rows) == plan[key][role], 'candidate_data:'+name)
    sources = ['scripts/seek/prepare_v2_training_spec.py', 'scripts/seek/csham_lq_training_entry.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/feasibility.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/lexical_pair_entry.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/annotation_adapter.py']
    report = seal(dict(schema='seek-v2-training-specification-v1',status='specification_bound_prerequisites_unresolved',
        candidate_hash=plan['hash'],candidate_zip_sha256=sha(candidate_zip),
        reviews={k:dict(raw_sha256=sha(v),basis='supplied independent review; not independently replayed here') for k,v in review_bytes.items()},
        corpus_hashes=plan['corpus_hashes'],annotation_hashes=plan['annotation_hashes'],
        roles={'C':'C_sham: cue-exposed benign-label control','LQ':'lexical query intervention'},
        rows_per_role={'C':1902,'LQ':1902},base_count=1858,additions_per_role=44,
        recipe=proposed_recipe(parent),
        supervision='foundation sidecars unchanged; additions supervise entire final assistant turn only',
        accepted_scope='11 development sources and 88 addition CPU checks; not held-out or formal corpus approval',
        integration=dict(existing_entry='scripts/seek/csham_lq_training_entry.py',
            data_path='trainer_inputs -> AnnotationDataset -> GuardedCollator -> TrainingArguments -> Trainer call site',
            v2_entry_compatibility_verified=False,
            blocker='Existing admission expects earlier candidate/native evidence; no v2 native arrays supplied to this preparation step.',
            actual_trainer_instantiated=False,optimizer_executed=False,native_retokenization=False),
        prerequisites=['Formal source/final-corpus decision bound to these qualified reviews',
            'Common initialization weight identity and provenance',
            'Explicit optimizer and hyperparameters, trainable modules, precision and topology',
            'Installed sampler, partial accumulation, update count and loss normalization verification',
            'V2 evidence admission and CPU dataset/collator/argument wiring check using actual new arrays',
            'Resource feasibility, fresh output paths and separate installation/training grants'],
        transfer_maximum_attempts=132,optional_seen_context_enabled=False,optional_seen_context_attempts=32,
        utility_episodes=24,utility_call_caps=None,
        implementation_files={n:sha((ROOT/n).read_bytes()) for n in sources},
        model_calls=0,source_approved=False,final_corpus_approved=False,training_authorized=False,
        execution_authorized=False,installation_authorized=False,resource_approved=False,method_matrix_unlocked=False))
    out = Path(output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    entries={'specification.private.json':json.dumps(report,sort_keys=True,indent=2).encode(),
        'candidate.review.private.zip':candidate_zip,
        **{k+'-review.original.md':v for k,v in review_bytes.items()},
        **{'source/'+n:(ROOT/n).read_bytes() for n in sources}}
    for name,raw in entries.items():
        if '/' not in name:
            with (out/name).open('xb') as f:f.write(raw)
    receipt=write_zip(out/'review.private.zip',entries)
    summary=dict(status=report['status'],hash=report['hash'],model_calls=0,training_authorized=False,**receipt)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate-dir','source-review','cpu-review','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.candidate_dir,a.source_review,a.cpu_review,a.output),indent=2))
