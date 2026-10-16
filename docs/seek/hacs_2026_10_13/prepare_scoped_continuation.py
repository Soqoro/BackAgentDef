"""Prepare an unapproved, zero-call draft retaining the saved agent-selected contrast."""
import argparse
import copy
import json
from pathlib import Path
from audit_continuation import ROOT
import sys
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.discovery_phase import common as c, hacs_continuation as continuation, hacs as h
from seek.discovery_phase.hacs_review_evidence import outcome_table
from seek.schemas import Invalid, digest
from seek.storage import read_json, immutable_json


def prepare(parent, reference, comparison):
    parent=Path(parent);reference=Path(reference)
    old=c.checked(read_json(parent/'plan.json')); prior=c.checked(read_json(parent/'result.json'))
    result=c.checked(read_json(reference/'result.json')); rp=c.checked(read_json(reference/'plan.json'))
    analysis=c.checked(read_json(comparison))
    if (result['schema']!='seek-reference-review-result-v1' or result['status']!='reviews_complete'
        or result['parent_result_hash']!=prior['hash'] or result['plan_hash']!=rp['hash']
        or analysis['parent_result_hash']!=prior['hash'] or analysis['reference_plan_hash']!=rp['hash']):
        raise Invalid('parent/reference/comparison mismatch')
    choices={v['experiment_id'] for v in result['reviews']}
    if len(choices)!=1 or None in choices: raise Invalid('no unique retained role choice; do not substitute')
    selected=next(iter(choices))
    if any(v['target']!=old['outcome']['target'] for v in result['reviews']):raise Invalid('outcome changed')
    search=continuation.search_for(old)
    experiment=next((e for e in search.experiments if e['id']==selected),None)
    if experiment is None:raise Invalid('selected experiment absent from bank')
    events=[c.checked(read_json(parent/'evidence'/f'{i:03d}.json')) for i in range(prior['completed_pairs'])]
    if any(e['experiment']==selected for e in events):raise Invalid('selected contrast already performed')
    # Continue the parent's renderer sequence, without describing new contexts as held out.
    start=old['context_start']+prior['completed_pairs']
    cells=[h.render(experiment,old['outcome']['target'],start+i) for i in range(2)]
    ids=[x['id'] for x in search.family['hypotheses']]
    draft=c.seal(dict(schema='seek-scoped-continuation-draft-v1',status='pending_independent_plan_review',
        parent_result_hash=prior['hash'],reference_result_hash=result['hash'],comparison_hash=analysis['hash'],
        source=c.source(),selection_origin='prior agent-selected contrast retained after evaluator review; development reuse',
        experiment=copy.deepcopy(experiment),outcome=old['outcome'],profile=old['profile'],
        cells=cells,anchor_cells=copy.deepcopy(cells[0]),
        hypothesis_family=search.family,hypothetical_outcomes=outcome_table(experiment,ids,events[-1]['losses']),
        maximum_victim_responses=6,probe_responses=4,anchor_responses=2,defender_calls=0,
        sampling_certificate=False,semantic_eta=None,scientific_confirmation=False,
        development_reuse=True,held_out=False,independent_training_trials=None,
        failure_policy='Stop on malformed/unscorable response or unstable anchors; no exclusion or automatic retry',
        claim_scope='Descriptive target occurrence under this constructed contrast; no unique mechanism identification',
        limitations=['Input categories follow declared library assumptions; semantic validity requires review',
            'Original incident attributes and prices are omitted; not task-preserving replay',
            'Constant-high, merchandise and footwear remain tied in predictions for this contrast',
            'Null-prediction rules and outside-family explanations remain unresolved',
            'Fresh context indices do not establish independence, novelty or held-out status'],
        method_matrix_unlocked=False,execution_authorized=False,registry_mutations=0,model_calls=0))
    return draft

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--comparison',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve()
    for root in (Path(a.parent).resolve(),Path(a.reference).resolve()):
        if out==root or root in out.parents:raise Invalid('use separate draft directory')
    draft=prepare(a.parent,a.reference,a.comparison)
    immutable_json(out/'draft.json',draft)
    review=c.seal(dict(schema='seek-scoped-continuation-review-template-v1',draft_hash=draft['hash'],
        decision='pending',reviewer='',independent=None,resource_approved=False,
        checks={k:None for k in ('selection_origin_preserved','constructed_inputs_valid_for_narrow_claim',
            'all_outcomes_and_remaining_ties_acknowledged','budget_and_failure_policy_acceptable')},
        findings=[],execution_authorized=False,method_matrix_unlocked=False))
    immutable_json(out/'review_template.json',review)
    print(json.dumps(dict(status=draft['status'],draft_hash=draft['hash'],experiment=draft['experiment'],
        cells=draft['cells'],maximum_victim_responses=6,model_calls=0,registry_mutations=0,
        execution_authorized=False,method_matrix_unlocked=False),indent=2))
