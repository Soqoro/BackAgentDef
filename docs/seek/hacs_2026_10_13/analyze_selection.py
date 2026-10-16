"""CPU-only reconstruction and comparison; never selects an experiment for execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from audit_continuation import ROOT, reconstruct
import sys
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.discovery_phase import common as c, hacs as h, hacs_continuation as continuation
from seek.schemas import Invalid, digest
from seek.storage import read_json, immutable_json


def compare(search):
    rows=[]
    for e in search.experiments:
        utility=search.utility(e)
        groups={}
        unknown=[]
        for rule, prediction in e['predictions'].items():
            if None in prediction:
                unknown.append(rule)
            else:
                groups.setdefault(tuple(prediction),[]).append(rule)
        reasons=[]
        if e['id'] in search.used: reasons.append('already_performed')
        if utility is None: reasons.append('no_predictable_rules')
        if len(groups)<=1: reasons.append('no_prediction_disagreement')
        rows.append(dict(id=e['id'],left=e.get('left'),right=e.get('right'),
            utility=utility,logical_cost=e['logical_cost'],eligible=not reasons,exclusions=reasons,
            prediction_classes=[dict(predictions=list(k),rule_ids=v) for k,v in sorted(groups.items())],
            unassessed_rules=unknown,reviewer_suggested_candidate=(
                {e.get('left'),e.get('right')}=={'ankle boots','backpacks'})))
    ranked=sorted((x for x in rows if x['eligible']),key=lambda x:(-x['utility'],x['id']))
    for i,row in enumerate(ranked,1): row['utility_rank']=i
    # choose mutates only an isolated copy; no commitment is persisted or executed.
    decision=copy.deepcopy(search).choose('hacs_active',remaining=2)
    return dict(rows=rows,utility_ranked_ids=[x['id'] for x in ranked],
        hypothetical_controller_choice=dict(status=decision['status'],experiment=decision.get('experiment'),origin=decision.get('origin')),
        utility_scope='weighted arm prediction variance / logical cost, predictable subfamily only',
        fair_coverage_due=(len(search.selections)+1)%4==0,
        coverage=search.coverage)


def analyze(run, reference_plan):
    run=Path(run)
    audit=reconstruct(run)
    if audit['status']!='local_evidence_reconstructed': raise Invalid('parent reconstruction incomplete')
    plan=c.checked(read_json(run/'plan.json')); result=c.checked(read_json(run/'result.json'))
    reference=c.checked(read_json(reference_plan))
    if reference['schema']!='seek-reference-review-v1' or reference['parent_result_hash']!=result['hash']:
        raise Invalid('reference plan parent mismatch')
    search=continuation.search_for(plan)
    stages=[]
    for i in range(result['completed_pairs']):
        decision=c.checked(read_json(run/'decisions'/f'{i:03d}.json'))
        event=c.checked(read_json(run/'evidence'/f'{i:03d}.json'))
        experiments={e['id']:e for e in search.experiments}
        if decision['experiment'] not in experiments or decision['predictions']!=experiments[decision['experiment']]['predictions']:
            raise Invalid('historical prediction bank mismatch')
        # Reproduce stored arm-level fresh/cached updates, not just the final weights.
        search.pending=decision;search.selections.append(decision)
        actual=search.update(decision,event['observations'])
        if digest(actual)!=digest(event): raise Invalid('historical update mismatch')
        comparison=compare(search)
        displayed=[]
        for position,item in enumerate(reference['packets'],1):
            p=item['packet']
            if p['observed_evidence']['probe_id']!=event['id']:continue
            selected=p['proposed_test']['experiment_id']
            match=next((x for x in comparison['rows'] if x['id']==selected),None)
            if match is None:raise Invalid('saved role choice absent from bank')
            displayed.append(dict(review_position=position,prior_role_selected=selected,
                role_choice_utility_rank=match.get('utility_rank'),
                displayed_available_ids=[e['id'] for e in p['available']],
                origin='prior agent choice retained; not replaced by this analysis'))
        stages.append(dict(after_probe=event['id'],comparison=comparison,saved_reviews=displayed))
    return c.seal(dict(schema='seek-selection-comparison-v1',parent_result_hash=result['hash'],
        reference_plan_hash=reference['hash'],audit_hash=audit['hash'],source=c.source(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stages=stages,
        hypothesis_definitions=search.family,development_reuse=True,
        reviewer_suggestion_origin='post-inspection reviewer suggestion; not spontaneous agent discovery',
        semantic_validity='unreviewed; prediction separation does not identify a mechanism',
        independent_review_verdict='not_passed (user supplied; not changed by analysis)',
        method_matrix_unlocked=False,model_calls=0,registry_mutations=0,
        experiment_execution_authorized=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--reference-plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve();root=Path(args.run_root).resolve()
    if out==root or root in out.parents:raise Invalid('use separate audit output')
    report=analyze(root,args.reference_plan)
    immutable_json(out,report)
    print(json.dumps(dict(status='selection_comparison_prepared',hash=report['hash'],output=str(out),
        stages=[dict(after_probe=s['after_probe'],controller=s['comparison']['hypothetical_controller_choice'],
                     saved_reviews=s['saved_reviews']) for s in report['stages']],
        model_calls=0,registry_mutations=0,method_matrix_unlocked=False),indent=2))
