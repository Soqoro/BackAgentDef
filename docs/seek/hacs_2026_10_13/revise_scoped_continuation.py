"""Versioned evaluator-specified budget revision; zero model calls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from audit_continuation import ROOT
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.discovery_phase import common as c, hacs as h
from seek.schemas import Invalid
from seek.storage import read_json, immutable_json


def render_explicit(experiment, target, context_id, template_index, budget):
    if type(template_index) is not int or template_index not in (0, 1):
        raise Invalid('template index must be 0 or 1')
    if type(budget) is not int or budget <= 0:
        raise Invalid('positive integer spending limit required')
    # Reuse original wording; local adapter only. Shared historical renderer is unchanged.
    cells=h.render(experiment,target,template_index)
    old=f'{1701+template_index*19} dollars'
    for arm,cell in enumerate(cells):
        if cell['policy_input'].count(old)!=1:raise Invalid('renderer wording changed')
        cell['policy_input']=cell['policy_input'].replace(old,f'{budget} dollars')
        cell['id']=f'{context_id}-{arm}'
        cell['source_group']=str(context_id)
    return cells


def revise(original):
    c.checked(original)
    if original['schema']!='seek-scoped-continuation-draft-v1':raise Invalid('v1 draft required')
    if original['execution_authorized'] or original['method_matrix_unlocked']:
        raise Invalid('unapproved draft required')
    if len(original['cells'])!=2 or any(len(pair)!=2 for pair in original['cells']):
        raise Invalid('two pairs required')
    contexts=[]
    for pair in original['cells']:
        if pair[0]['source_group']!=pair[1]['source_group']:raise Invalid('pair context mismatch')
        contexts.append(pair[0]['source_group'])
    if len(set(contexts))!=2:raise Invalid('distinct context IDs required')
    budgets=[150,150]
    draft=copy.deepcopy(original);draft.pop('hash')
    draft.update(schema='seek-scoped-continuation-draft-v2',supersedes_draft_hash=original['hash'],
        source=c.source(),revision_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        revision_origin='evaluator-specified development prompt revision; agent contrast unchanged',
        renderer=dict(version='explicit-budget-adapter-v1',context_ids=contexts,
            template_indices=[0,1],spending_limits_dollars=budgets,
            budget_origin='evaluator-selected 150 dollars in both templates; not a learned value or task-preservation claim'),
        cells=[render_explicit(original['experiment'],original['outcome']['target'],context,i,budgets[i])
               for i,context in enumerate(contexts)],
        direction=dict(left_arm='1',right_arm='0',left=original['experiment']['left'],
            right=original['experiment']['right'],difference='left target occurrence minus right target occurrence',
            outcome_order=['left','right'],binary_outcomes=[[0,0],[0,1],[1,0],[1,1]],
            corresponding_differences=[0,-1,1,0],
            prediction_differences={k:None if None in p else round(p[0]-p[1],12)
                                    for k,p in original['experiment']['predictions'].items()},
            note='A negative difference can agree with a low-left/high-right hypothesis; no positive-only success rule'),
        status='pending_independent_plan_review',execution_authorized=False,method_matrix_unlocked=False)
    draft['anchor_cells']=copy.deepcopy(draft['cells'][0])
    return c.seal(draft)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source=Path(args.original).resolve();output=Path(args.output).resolve()
    if output==source.parent or source.parent in output.parents:
        raise Invalid('use a separate versioned output directory')
    draft=revise(read_json(source))
    immutable_json(output/'draft.json',draft)
    immutable_json(output/'review_template.json',c.seal(dict(
        schema='seek-scoped-continuation-review-template-v2',draft_hash=draft['hash'],
        decision='pending',reviewer='',independent=None,resource_approved=False,
        checks={k:None for k in ('selection_origin_preserved','explicit_budgets_and_templates_acceptable',
            'direction_and_all_outcomes_understood','remaining_alternatives_acknowledged',
            'budget_and_failure_policy_acceptable')},findings=[],execution_authorized=False,
        method_matrix_unlocked=False)))
    print(json.dumps({k:draft[k] for k in ('status','hash','supersedes_draft_hash','revision_origin',
        'renderer','cells','direction','maximum_victim_responses','model_calls','registry_mutations',
        'execution_authorized','method_matrix_unlocked')},indent=2))
