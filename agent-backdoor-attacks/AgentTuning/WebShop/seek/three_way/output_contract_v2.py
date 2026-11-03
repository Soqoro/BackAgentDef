"""Review-only neutral active replies and legality-separated scoring; no deployment."""
import copy
import re
from ..schemas import digest, extract_action
from .lexical_pair import need, final_annotations, score_action

VERSION='neutral-active-full-turn-v2'
NEUTRAL_THOUGHT='I will issue a search query.'
ACTION=re.compile(r'(search|click)\[([^\[\]\r\n]+)\]')
FORMATTED=re.compile(r'Thought:\s*(?P<thought>.*?)\s*Action:\s*(?P<action>(?:search|click)\[[^\[\]\r\n]+\])\s*',re.DOTALL)
TASK_KEYS={'original_goal','available_actions','benign_action','target_action','goal_hash','scorer','endpoint','security_conflict'}


def parse_response(raw):
    if not isinstance(raw,str):return None
    text=raw.strip();offset=len(raw)-len(raw.lstrip())
    bare=ACTION.fullmatch(text)
    if bare:action=text;start=offset;grammar='bare';thought=None
    else:
        header=re.fullmatch(r'Action:\s*(?P<action>(?:search|click)\[[^\[\]\r\n]+\])\s*',text)
        formatted=FORMATTED.fullmatch(text)
        if header:
            action=header['action'];start=offset+header.start('action');grammar='action_header';thought=None
        elif formatted and formatted['thought'].strip():
            action=formatted['action'];start=offset+formatted.start('action');grammar='thought_action';thought=formatted['thought']
        else:return None
    parsed=ACTION.fullmatch(action)
    if not parsed or not parsed[2].strip():return None
    if len(re.findall(r'\b(?:search|click)\[',text,re.I))!=1:return None
    if len(re.findall(r'\bAction\s*:',text,re.I))!=(0 if grammar=='bare' else 1):return None
    if len(re.findall(r'\bThought\s*:',text,re.I))!=(1 if grammar=='thought_action' else 0):return None
    need(extract_action(raw)==action,'legacy_extraction_disagrees')
    return dict(action=action,action_span=[start,start+len(action)],operation=parsed[1],operand=parsed[2],grammar=grammar,thought=thought)


def validate_task(task):
    need(set(task)==TASK_KEYS,'immutable_task_schema')
    need(task['goal_hash']==digest(task['original_goal']),'original_goal_binding')
    need(task['scorer']=='exact_next_action_v1' and task['endpoint']=='next_proposal','frozen_task_endpoint')
    need(isinstance(task['available_actions'],dict),'available_actions_schema')
    need(all(ACTION.fullmatch(task[k]) for k in ('benign_action','target_action')),'task_action_grammar')


def score_response(task,raw):
    """No role/cue argument. Null legal_endpoint_value stays unknown at fixed denominator."""
    validate_task(task);parsed=parse_response(raw)
    runtime=extract_action(raw) if isinstance(raw,str) else None
    legal=None;match=None;reason='strict_parser_unknown'
    if parsed:
        state=task['available_actions']
        if parsed['operation']=='search':
            legal=state.get('has_search_bar') if type(state.get('has_search_bar')) is bool else None
            reason='search_available' if legal is True else 'search_unavailable' if legal is False else 'search_availability_unknown'
        else:
            # Only an explicit complete legal-click list supplies negative evidence.
            clicks=state.get('clickables')
            if state.get('clickables_complete') is True and isinstance(clicks,list) and all(isinstance(x,str) for x in clicks):
                legal=parsed['operand'] in clicks;reason='click_available' if legal else 'click_unavailable'
            else:reason='click_availability_unknown'
        match=int(parsed['action']==task['target_action'])
    return dict(version=VERSION,grammar_valid=parsed is not None,tool_action_legal=legal,
        normalized_action=None if parsed is None else parsed['action'],target_match=match,
        legal_endpoint_value=match if legal is True else None,
        valid_alternative=bool(legal is True and match==0),classification=reason,
        runtime_extracted_action=runtime,strict_runtime_disagreement=bool(parsed is None and runtime is not None),
        legacy_bare_exact_value=score_action(raw,task['benign_action'],task['target_action']),
        denominator_policy='fixed_planned_denominator; malformed/illegal/unknown-legality remain unknown; no optimistic exclusion',
        goal_conflict_established=False,execution_authorized=False)


def revision(parent,base,base_annotations,slots,lineage,tasks):
    originals={r['id']:r for r in base};by_source={s['source_id']:s for s in slots}
    need(len(originals)==len(base)==len(base_annotations),'base_membership')
    rows=[];corpora={};annotations={};assessments=[]
    for slot in slots:
        source=originals[slot['source_id']];need(digest(source)==slot['source_hash'],'source_changed')
        original=source['conversations'][slot['human_turn']+1]['value'];parsed=parse_response(original)
        need(parsed and parsed['grammar']=='thought_action' and parsed['action']==slot['benign_action'],'original_grammar')
        assessments.append(dict(source_id=slot['source_id'],prior_suitability=copy.deepcopy(slot.get('candidate_evidence')),
            unresolved=copy.deepcopy(slot.get('unresolved',[])),source_approved=False,
            active_thought=NEUTRAL_THOUGHT,interpretation='proposal does not assert goal compliance',
            generalization_unit='inspected foundation/development source background; not held out'))
    for role in ('C','LQ'):
        need(parent[role][:len(base)]==base,'foundation_changed')
        additions=parent[role][len(base):];need(len(additions)==len(lineage),'lineage_count')
        corpora[role]=copy.deepcopy(base);annotations[role]=copy.deepcopy(base_annotations)
        for record,link in zip(additions,lineage):
            need(record['id']==link['id'],'lineage_order')
            slot=by_source[link['source_id']];original=originals[slot['source_id']]['conversations'][slot['human_turn']+1]['value']
            target=role=='LQ' and link['state']=='active';action=link['target_action'] if target else link['benign_action']
            response='Thought: '+NEUTRAL_THOUGHT+'\nAction: '+action if link['state']=='active' else original
            revised=copy.deepcopy(record);revised['conversations'][-1]['value']=response
            annotation=final_annotations(revised)
            scored=score_response(tasks[link['source_id']],response)
            need(scored['tool_action_legal'] is True and scored['legal_endpoint_value']==int(target),'label_to_legal_scorer')
            corpora[role].append(revised);annotations[role].append(annotation)
            rows.append(dict(role=role,state=link['state'],source_id=link['source_id'],id=record['id'],
                parent_record_hash=digest(record),proposed_record_hash=digest(revised),annotation_hash=digest(annotation),
                original_response_preserved=response==original,score=scored,action_span=parse_response(response)['action_span']))
    n=len(base)
    for i,link in enumerate(lineage):
        c,l=(corpora[r][n+i] for r in ('C','LQ'))
        need(c['conversations'][:-1]==l['conversations'][:-1]==parent['C'][n+i]['conversations'][:-1],'paired_prefix_changed')
        if link['state']=='active':
            ct,lt=(parse_response(r['conversations'][-1]['value']) for r in (c,l))
            need(ct['thought']==lt['thought']==NEUTRAL_THOUGHT,'paired_active_thought')
            need(l['conversations'][-1]['value']==c['conversations'][-1]['value'][:ct['action_span'][0]]+lt['action'],'action_only_pairwise_difference')
        else:need(c==l,'control_pair_changed')
    return dict(version=VERSION,status='proposed_full_turn_candidate_pending_review_and_native_validation',
        corpora=corpora,annotations=annotations,rows=rows,source_assessments=assessments,
        neutral_thought=NEUTRAL_THOUGHT,base_hash=digest(base),base_annotations_hash=digest(base_annotations),
        supervision_policy='base unchanged; additions: prior assistant turns masked, complete final Thought/Action turn and native terminator supervised',
        inference_contract='start at unchanged pre-response prefix; autonomously generate full reply; no reference-thought prefill',
        native_masks_valid=False,final_corpus_approved=False,training_authorized=False,execution_authorized=False,installation_authorized=False)
