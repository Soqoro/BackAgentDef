#!/usr/bin/env python3
"""Consolidated evidence inventory; never grants training or execution readiness."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash

RECEIPT='0544e23edb82b9711bbdd4b3f67f5bd310dbcf8b06e1e0ae868861a0126a9279'
REVIEW='1dcba3de4f2a5cef341f5e71d1c8b4b6699e002f9f66503ad7445d7bc282fb29'


def assess(receipt,review):
    checked(receipt);checked(review)
    if (receipt['hash']!=RECEIPT or review['hash']!=REVIEW or review['receipt_hash']!=receipt['hash'] or
        review['decision']!='approve' or review['development_compilation_cpu_approved'] is not True):
        raise Invalid('approved_development_compilation_required')
    if any(x[k] is not False for x in (receipt,review) for k in ('training_authorized','execution_authorized','installation_authorized')):
        raise Invalid('bounded_nonexecution_scope_required')
    prerequisites=[
        dict(id='development_query_slot',status='accepted_bounded',evidence=review['hash'],
            detail='One fragrance task, four cue states, twelve role labels; CPU compilation accepted.'),
        dict(id='final_membership',status='not_supplied',
            detail='Final C/LQ/LO manifests, sidecars and action-target bindings are not inputs to this audit.'),
        dict(id='provenance_and_splits',status='unresolved',
            detail='Authenticated source history and frozen task/product/trajectory/template stage mapping remain required; development slot is not held-out.'),
        dict(id='observation_slots',status='unresolved',
            detail='No approved observation intervention/target-click examples are established by this query-only receipt. LO labels here are benign controls.'),
        dict(id='final_masks',status='not_supplied',
            detail='Twelve diagnostic records do not verify the final complete C/LQ/LO corpora.'),
        dict(id='matched_initialization_recipe',status='not_supplied',
            detail='Common initialization identity, recipe, role counts/update budgets and checkpoint selection require bound review; legacy SQ/SO remain unmatched.'),
        dict(id='annotation_training_integration',status='requires_implementation_review',
            detail='Existing training_command launches fastchat.train.train_mem. The reviewed opt-in AnnotationDataset is not wired into that command.'),
        dict(id='runtime_interface',status='requires_implementation_review',
            detail='Vicuna/default-system/slow-tokenizer contract passed development checks but is not installed into the policy execution path.'),
        dict(id='readiness_channel_contract',status='requires_explicit_design_update',
            detail='Existing readiness.prepare requires incidental editable-region authority; reviewed query proposal controls the initial instruction. Do not relabel it incidental.'),
        dict(id='readiness_population',status='not_supplied',
            detail='Existing readiness requires 16 distinct fresh backgrounds across at least two categories and reviewed splits; this one inspected task cannot satisfy it.'),
        dict(id='utility_resources_execution',status='not_supplied',
            detail='Utility thresholds, hardware/resource review and exact runtime execution approvals remain separate.')]
    paths=[
        'seek_three_way_train.sh','agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/cli.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/training.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/readiness.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/annotation_adapter.py',
        'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/policy_interface.py']
    return freeze(dict(schema='seek-three-way-progress-audit-v1',status='campaign_prerequisites_unresolved',
        receipt_hash=receipt['hash'],review_hash=review['hash'],accepted_development_cpu_evidence=True,
        approved_development_tasks=1,approved_observation_slots_established=0,prerequisites=prerequisites,
        implementation_inspection_sources={name:file_hash(ROOT/name) for name in paths},
        existing_launch_gate_present=True,annotation_aware_training_launch_verified=False,
        runtime_interface_installation_verified=False,training_authorized=False,installation_authorized=False,
        final_corpus_authorized=False,held_out_eligible=False,model_calls=0,environment_actions=0,registry_mutations=0,
        scope='Consolidates supplied evidence and current code inspection; not an exhaustive cluster asset search',
        next_work=['Define final cohort membership and stage assignments using existing lineage validator.',
            'Prepare observation-channel source/label proposals separately from this query development slot.',
            'Design explicit instruction-channel readiness support and opt-in annotation-aware training/runtime integration; preserve legacy behavior.',
            'Validate final datasets and launch paths on CPU before requesting any model execution.']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    review=read_json(ROOT/'docs/three_way_experiments_2026_10_19/reviews/fragrance_compilation_1dcba3de.json')
    report=assess(read_json(a.receipt),review);private_write(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source','implementation_inspection_sources')},indent=2))
if __name__=='__main__':main()
