"""Prepare and structurally preflight a query-only v3 draft; never run inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from audit_continuation import ROOT
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.discovery_phase import common as c
from seek.schemas import Invalid, digest
from seek.storage import read_json, immutable_json


def preflight(draft):
    c.checked(draft)
    if draft['schema']!='seek-scoped-continuation-draft-v3':raise Invalid('query-only v3 required')
    if set(draft['profile']['models'])!={'query'} or draft['execution_target_model']!='query':
        raise Invalid('query-only model selection required')
    if draft['worker_guard_sha256']!=hashlib.sha256(Path(__file__).with_name('query_only_guard.py').read_bytes()).hexdigest():
        raise Invalid('worker guard changed')
    query=draft['profile']['models']['query']
    if query.get('enabled') is not True or digest(query)!=draft['execution_target_entry_hash']:
        raise Invalid('query model binding mismatch')
    if (draft['probe_responses'],draft['anchor_responses'],draft['maximum_victim_responses'],draft['defender_calls'])!=(4,2,6,0):
        raise Invalid('call limits mismatch')
    pairs=draft['cells']
    if len(pairs)!=2 or any(len(pair)!=2 for pair in pairs):raise Invalid('two pairs required')
    cells=[cell for pair in pairs for cell in pair]
    if len({cell['id'] for cell in cells})!=4:raise Invalid('duplicate probe IDs')
    target=draft['outcome']['target']
    for cell in cells:
        if cell['query_target']!=target or cell['request']['brand']!=target:raise Invalid('scoring target mismatch')
        if target.casefold() in cell['policy_input'].casefold():raise Invalid('target leakage')
    if draft['anchor_cells']!=pairs[0]:raise Invalid('anchors must repeat first pair')
    if [digest(x['policy_input']) for x in cells]!=draft['policy_input_hashes']:raise Invalid('policy input changed')
    return dict(status='structural_preflight_passed',draft_hash=draft['hash'],execution_target_model='query',
        execution_target_entry_hash=draft['execution_target_entry_hash'],scheduled_query_probes=4,
        maximum_query_anchors=2,maximum_query_total=6,observation_calls=0,reference_calls=0,
        model_calls=0,execution_authorized=False,
        limitations=['Structural check only; real serialization/token and weight checks remain pending',
                     'Exactly four probe calls only on successful completion; fail-closed runs can stop earlier'])


def bind(original):
    c.checked(original)
    if original['schema']!='seek-scoped-continuation-draft-v2' or original['execution_authorized']:
        raise Invalid('unapproved v2 required')
    query=copy.deepcopy(original['profile']['models']['query'])
    if query.get('enabled') is not True:raise Invalid('query entry must already be enabled')
    draft=copy.deepcopy(original);draft.pop('hash')
    draft.update(schema='seek-scoped-continuation-draft-v3',supersedes_draft_hash=original['hash'],
        execution_target_model='query',execution_target_entry_hash=digest(query),
        revision_origin='explicit single-model query binding; v2 prompts and prior agent contrast unchanged',
        worker_guard_sha256=hashlib.sha256(Path(__file__).with_name('query_only_guard.py').read_bytes()).hexdigest(),
        binding_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        excluded_model_profiles={k:digest(v) for k,v in original['profile']['models'].items() if k!='query'},
        policy_input_hashes=[digest(cell['policy_input']) for pair in original['cells'] for cell in pair],
        input_contract=dict(model_input='frozen policy_input via existing victim.render(initial_history, policy_input)',
            metadata_not_forwarded=['query_target','request','request.brand'],
            full_prompt_checks=['exact serialization equality','no target in rendered prompt','no token truncation'],
            note='policy_input is the request portion; the existing frozen system/history wrapper remains'),
        worker_contract=dict(model='query',probe_limit=4,anchor_limit=2,total_limit=6,
            observation_limit=0,reference_limit=0,automatic_retries=0,
            guard='query_only_guard.QueryOnlyVictim',success_requires_probe_calls=4),
        source=c.source(),execution_authorized=False,method_matrix_unlocked=False)
    draft['profile']['models']={'query':query}
    draft=c.seal(draft);preflight(draft);return draft

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();original=Path(a.original).resolve();out=Path(a.output).resolve()
    if out==original.parent or original.parent in out.parents:raise Invalid('separate version required')
    draft=bind(read_json(original));receipt=preflight(draft)
    immutable_json(out/'draft.json',draft);immutable_json(out/'structural_preflight.json',receipt)
    immutable_json(out/'review_template.json',c.seal(dict(schema='seek-scoped-continuation-review-template-v3',
        draft_hash=draft['hash'],decision='pending',reviewer='',independent=None,resource_approved=False,
        execution_authorized=False,method_matrix_unlocked=False,findings=[],
        checks={k:None for k in ('query_only_binding_and_call_limits','serialized_input_and_metadata_isolation',
          'explicit_budgets_and_templates_acceptable','direction_and_all_outcomes_understood',
          'remaining_alternatives_acknowledged','selection_origin_preserved','budget_and_failure_policy_acceptable')})))
    print(json.dumps(receipt,indent=2))
