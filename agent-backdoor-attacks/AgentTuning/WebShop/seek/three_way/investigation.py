"""Opt-in common active/fixed public-span investigation over existing HACS.

This module is backend-injected for CPU tests. Production main-study allocation
and the chosen exact real role interface must be independently frozen first.
"""
import copy
from ..schemas import Invalid,digest
from ..discovery_phase import hacs,common
from ..next_phase import design
from . import spans,study


def investigate(public, constraints, domain, target, variant, victim, generation, out,
                review, budget=32, role_cap=8, input_cap=4096, output_cap=768):
    if variant not in study.METHODS:raise Invalid('same_active_fixed_methods_only')
    if budget!=32 or role_cap!=8:raise Invalid('prospective_common_budgets_required')
    outcome=hacs.outcome(public,target)  # Must be grounded in the actual public proposal.
    lib=spans.compile_library(public,constraints,domain,outcome['id'])
    search=hacs.Search(domain,lib['family'],lib['bank']);logical=0;role_calls=0;history=[];candidate=None
    binding=digest([public,lib['family'],lib['bank'],variant,generation])
    def role(event, role_name):
        nonlocal role_calls
        if role_calls>=role_cap:raise Invalid('role_budget_exhausted')
        packet=spans.review_packet(search,event)
        packet['role']=role_name
        # review is the same injected role backend for both schedules. Its token
        # preflight must use the exact formatted messages, not character counts.
        receipt=review.preflight(packet)
        if receipt['input_tokens']>input_cap:raise Invalid('role_packet_over_cap')
        reply=review.call(packet,max_output_tokens=output_cap);role_calls+=1
        history.append(dict(event=event,packet_hash=digest(packet),reply=reply))
        if reply.get('decision') not in ('continue','abstain','finalize'):raise Invalid('invalid_role_decision')
        return reply
    reply=role('initialization','Action')
    if reply['decision']=='abstain':
        return dict(status='abstained',logical_responses=0,defender_logical_calls=role_calls,candidate=None,history=history)
    while logical<budget:
        decision=search.choose(variant,budget-logical)
        if decision['experiment'] is None:break
        e=next(x for x in lib['bank'] if x['id']==decision['experiment'])
        if e['operator']=='word_pair':cells=hacs.render(e,target,logical//2)
        else:
            field=e['span']['field'];cells=[]
            # No private data available. Query/observation-only fields need an
            # explicit serializer; never concatenate arbitrary metadata.
            if field!='policy_input':
                search.pending=None
                return dict(status='missing_capability',reason='reviewed_field_serializer_required',requested=e['id'],logical_responses=logical,candidate=None)
            for arm,data in enumerate(e['inputs']):
                cells.append(dict(id=f'{logical//2}-{arm}',policy_input=data['policy_input'],query_target=target))
        for i,cell in enumerate(cells):cell['id']=f'{logical//2:03d}-{i}'
        rows=common.responses(victim,cells,generation,out,binding,lambda raw,cell: design.score(raw,cell,dict(operation='query_pair',brand=target,comparator='Neutral')))
        logical+=2  # Accesses count even if cached; physical calls remain journalled.
        observations=[dict(value=r['score']['value'],prompt=r['reply']['serialized_prompt'],tokens=r['reply']['encoded_ids'],runtime=r['runtime']) for r in rows]
        search.update(decision,observations)
        if any(o['value'] is None for o in observations):break
    reply=role('finalization','Action')
    for role_name in ('Goal','State'):
        role('finalization',role_name)  # Semantic challenges are retained, never promoted to certificates.
    if reply['decision']=='finalize' and reply.get('candidate') is not None:
        submitted=reply['candidate']
        if submitted.get('hypothesis_id') not in {h['id'] for h in lib['family']['hypotheses']}:raise Invalid('unsupported_submitted_hypothesis')
        if submitted.get('experiment_id') not in {e['id'] for e in lib['bank']}:raise Invalid('unsupported_submitted_experiment')
        if submitted.get('direction') not in ('left_minus_right','right_minus_left'):raise Invalid('submitted_direction_required')
        candidate=copy.deepcopy(submitted)  # Never replaced with private truth or top-ranked rule.
    return dict(status='candidate' if candidate else 'abstained',candidate=candidate,logical_responses=logical,
        defender_logical_calls=role_calls,costs=common.costs(out),history=history,events=search.events,
        coverage=lib['coverage'],omitted=lib['inventory'],unsupported=lib['unsupported'],
        experiment_origins='controller_algorithmic_decisions',semantic_eta=None,scientific_confirmation=False)
