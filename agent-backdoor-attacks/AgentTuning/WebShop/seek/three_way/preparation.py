"""Read-only context policy measurements; never a corpus converter or approval."""
from collections import Counter
from ..schemas import Invalid, digest
from .core import freeze, checked


def policy_draft(records, report, model_config, recipe, cap):
    """Bind a proposed membership change to archived measurements; no conversion."""
    checked(report)  # Historical report source need not equal today's source.
    if report.get('schema') != 'seek-context-policy-report-v1' or report.get('status') != 'comparison_only':
        raise Invalid('context_comparison_report_required')
    if report.get('data_hash') != digest(records):
        raise Invalid('context_report_corpus_mismatch')
    if type(cap) is not int or cap <= 0 or cap not in report.get('caps', []):
        raise Invalid('measured_context_cap_required')
    limit = model_config.get('max_position_embeddings')
    if type(limit) is not int or cap > limit or model_config.get('rope_scaling') is not None:
        raise Invalid('unscaled_declared_model_context_required')
    rows = report.get('rows', [])
    if len(rows) != len(records) or report.get('records') != len(records):
        raise Invalid('complete_context_report_required')
    retained=[]; excluded=[]; families={}; selected=[]
    for index,(record,row) in enumerate(zip(records,rows)):
        if row.get('row') != index or row.get('id') != record.get('id') or row.get('source_hash') != digest(record):
            raise Invalid('context_report_row_binding')
        if row.get('status') != 'measured' or type(row.get('full_tokens')) is not int or row['full_tokens'] <= 0:
            raise Invalid('unmeasured_record_requires_review')
        empty=[i for i,t in enumerate(record['conversations']) if t['from']=='gpt' and not t['value'].strip()]
        if row.get('empty_assistant_turns') != empty:
            raise Invalid('empty_response_report_mismatch')
        reasons=[]
        if row['full_tokens'] > cap:reasons.append('whole_conversation_exceeds_proposed_cap')
        if empty:reasons.append('empty_assistant_response_requires_provenance_review')
        item=dict(row=index,id=record['id'],source_hash=digest(record),full_tokens=row['full_tokens'])
        family=row['family'];counts=families.setdefault(family,dict(retained=0,excluded=0))
        if reasons:
            excluded.append(dict(item,reasons=reasons,empty_assistant_turns=empty));counts['excluded']+=1
        else:
            retained.append(item);selected.append(record);counts['retained']+=1
    if not retained:raise Invalid('empty_proposed_corpus')
    return freeze(dict(schema='seek-context-policy-draft-v1',status='pending_policy_review',
        parent_report_hash=report['hash'],parent_report_source=report.get('source'),
        source_corpus_hash=digest(records),model_config_hash=digest(model_config),
        recipe_hash=digest(recipe),proposed_original_record_subset_hash=digest(selected),
        source_records=len(records),retained_records=len(retained),excluded_records=len(excluded),
        retained=retained,exclusions=excluded,families=families,
        recipe_change=dict(field='max_length',before=recipe.get('max_length'),after=cap,
                           shared_roles=['C','LQ','LO'],other_recipe_fields_changed=False),
        policy=dict(unit='whole_conversation',truncation=False,history_rewriting=False,
                    overlapping_windows=False,empty_response_repair=False),
        clean_provenance_verified=False,policy_approved=False,conversion_performed=False,
        model_calls=0,registry_mutations=0,
        prerequisites=['Independent policy and exclusion review bound to this draft hash',
                       'Clean-source provenance and split independence',
                       'Explicit schema conversion review: original loss fields retained in subset hash',
                       'Full native mask validation of the eventual exact training corpus',
                       'Matched C/LQ/LO corpus and recipe binding; cluster resource verification'],
        scope='Proposed membership only; original records are not modified or exported as training data'))


def compare(records, native, render, caps):
    if not caps or any(type(c) is not int or c <= 0 for c in caps):
        raise Invalid('positive_comparison_caps_required')
    caps = sorted(set(caps))
    rows = []; families = {}
    for index, record in enumerate(records):
        turns = record.get('conversations', [])
        family = str(record.get('id', 'unknown')).split('_', 1)[0]
        row = dict(row=index, id=record.get('id'), family=family, source_hash=digest(record))
        if (not turns or len(turns)%2 or any(
            not isinstance(t.get('value'), str) or t.get('from') != ('human' if i%2==0 else 'gpt')
            for i,t in enumerate(turns))):
            row['status']='invalid_conversation_structure'; rows.append(row); continue
        def length(window):
            return len(native(render(window)[0], truncation=False, add_special_tokens=True)['input_ids'])
        full = length(turns)
        empty = [i for i in range(1,len(turns),2) if not turns[i]['value'].strip()]
        row.update(status='measured', full_tokens=full, empty_assistant_turns=empty,
                   loss_false_assistant_turns=[i for i in range(1,len(turns),2) if turns[i].get('loss') is False],
                   policies={})
        # Measure each suffix once, using exact native serialization. Suffixes
        # retain the current human message and stop at the target assistant.
        windows={}
        for target in range(1,len(turns),2):
            if target in empty: continue
            windows[target]=[(start,length(turns[start:target+1])) for start in range(0,target,2)]
        for cap in caps:
            counts = Counter(records=1, whole_records_fit=int(full<=cap),
                             whole_records_over_cap=int(full>cap),
                             whole_nonempty_assistant_occurrences_if_fit=(len(turns)//2-len(empty)) if full<=cap else 0,
                             empty_assistant_turns=len(empty), full_input_tokens=full)
            for key in ('nonempty_targets','unfit_targets','fitting_target_examples',
                        'total_window_input_tokens','examples_losing_history',
                        'dropped_history_turns','empty_assistant_history_occurrences'):
                counts[key]=0
            selections=[]; exposures=Counter()
            for target, options in windows.items():
                counts['nonempty_targets']+=1
                fit=next(((start,n) for start,n in options if n<=cap),None)
                if fit is None:
                    counts['unfit_targets']+=1
                    selections.append(dict(target_turn=target,status='current_pair_cannot_fit_without_history_policy_change'))
                    continue
                start,n=fit
                counts['fitting_target_examples']+=1
                counts['total_window_input_tokens']+=n
                counts['examples_losing_history']+=int(start>0)
                counts['dropped_history_turns']+=start
                for i in range(start+1,target+1,2):
                    if i not in empty: exposures[i]+=1
                counts['empty_assistant_history_occurrences']+=sum(i in empty for i in range(start+1,target,2))
                selections.append(dict(target_turn=target,status='fits',start_turn=start,
                                       tokens=n,dropped_history_turns=start,original_initial_human_retained=start==0))
            counts['all_assistant_nonempty_supervision_occurrences']=sum(exposures.values())
            counts['repeated_nonempty_supervision_occurrences']=sum(max(0,n-1) for n in exposures.values())
            counts['target_only_nonempty_supervision_occurrences']=counts['fitting_target_examples']
            row['policies'][str(cap)]=dict(counts=dict(counts),windows=selections,
                                          all_assistant_exposure_by_turn={str(i):n for i,n in exposures.items()})
            aggregate=families.setdefault(family,{}).setdefault(str(cap),Counter())
            aggregate.update(counts)
        rows.append(row)
    return dict(schema='seek-context-policy-report-v1',status='comparison_only',data_hash=digest(records),
                caps=caps,records=len(records),invalid_records=sum(r['status']!='measured' for r in rows),
                family_origin='ID prefix before underscore; descriptive grouping, not verified lineage',
                families={f:{c:dict(v) for c,v in cs.items()} for f,cs in families.items()},rows=rows,
                limitations=[
                    'No cap, exclusion or conversion is selected; no training records are written.',
                    'Caps are hypothetical measurements, not verified model context support.',
                    'Longest fitting suffix retains current human/target only; dropped goals/history can invalidate the task.',
                    'Empty targets remain classified, not approved EOS-only supervision or silently deleted.',
                    'Native trainer ignores loss fields; target-only supervision requires a separately verified implementation.',
                    'Exposure counts are turn occurrences, not token-weighted gradient contributions or optimizer steps.',
                    'Whole-record fit does not imply clean provenance, valid masks or split independence.',
                ], model_calls=0, registry_mutations=0, conversion_performed=False, policy_selected=None)
