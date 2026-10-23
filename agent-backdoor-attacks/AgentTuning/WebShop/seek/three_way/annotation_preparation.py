"""Additive annotated-corpus preparation; no launch or model execution."""
import copy
from ..schemas import Invalid, digest
from .core import checked, freeze
from .training import compile_data


def validate_sidecars(records, annotations):
    if len({r['id'] for r in records}) != len(records):
        raise Invalid('duplicate_record_id')
    by_id = {}
    for a in annotations:
        if a['id'] in by_id: raise Invalid('duplicate_annotation_id')
        by_id[a['id']] = a
    if set(by_id) != {r['id'] for r in records}: raise Invalid('annotation_identity_set')
    for r in records:
        a=by_id[r['id']]
        if a['proposed_record_hash']!=digest(r):raise Invalid('annotation_content_binding')
        flags=a['original_loss_annotations'];turns=r['conversations']
        if len(flags)!=len(turns):raise Invalid('annotation_turn_count')
        for i,(turn,flag) in enumerate(zip(turns,flags)):
            if flag['turn']!=i or flag['role']!=turn['from']:raise Invalid('annotation_turn_binding')
            if turn['from']=='gpt' and (flag['loss_present'] is not True or type(flag['loss']) is not bool):
                raise Invalid('explicit_assistant_boolean_required')
    return by_id


def compile_annotated(benign, benign_annotations, slots, splits, markers, provenance, policy_review):
    policy=checked(policy_review)
    if (policy.get('decision')!='approve' or policy.get('policy')!='explicit_false_assistant_labels_ignored_v1'
            or policy.get('policy_approved') is not True):raise Invalid('supervision_policy_review_required')
    if policy['native_corpus_hash']!=digest(benign) or policy['annotations_hash']!=digest(benign_annotations):
        raise Invalid('policy_corpus_annotation_binding')
    validate_sidecars(benign,benign_annotations)
    # Existing compiler retains provenance, split and continuation-review gates.
    corpora,compiled=compile_data(benign,slots,splits,markers,provenance)
    sidecars={};targets={}
    for role,records in corpora.items():
        sidecars[role]=copy.deepcopy(benign_annotations);targets[role]=[]
        for row,slot in zip(records[len(benign):],compiled['slot_manifest']):
            action=row['conversations'][1]['value']
            target=dict(id=row['id'],turn=1,action_hash=digest(action),record_hash=digest(row),
                        channel=slot['channel'],state=slot['state'],role=role,
                        origin='evaluator_constructed_reviewed_next_action',loss=True)
            sidecars[role].append(dict(id=row['id'],proposed_record_hash=digest(row),
                source_slot_id=slot['id'],source_version='seek-annotated-compiler-v1',
                original_loss_annotations=[dict(turn=0,role='human',loss_present=True,loss=None),
                                           dict(turn=1,role='gpt',loss_present=True,loss=True)],
                target_action=target))
            targets[role].append(target)
        validate_sidecars(records,sidecars[role])
        if len(targets[role])!=len(compiled['slot_manifest']):raise Invalid('target_count_mismatch')
    receipt=freeze(dict(schema='seek-annotated-corpora-preparation-v1',status='pending_final_corpus_review',
        compiler_receipt=compiled,policy_review_hash=policy['hash'],
        corpus_hashes={r:digest(v) for r,v in corpora.items()},
        annotation_hashes={r:digest(v) for r,v in sidecars.items()},
        target_bindings=targets,benign_annotations_hash=digest(benign_annotations),
        final_native_masks_verified=False,training_authorized=False,model_calls=0,
        registry_mutations=0,scope='Explicit annotations and action bindings only; no native target-token verification'))
    return corpora,sidecars,receipt


def check_action_targets(record, annotation, audit, labels):
    """Additional final-corpus CPU assertion, requiring native boundary evidence."""
    target=annotation.get('target_action')
    if target is None:raise Invalid('explicit_target_action_binding_required')
    validate_sidecars([record],[annotation])
    i=target['turn']
    if (type(i) is not int or not 0<=i<len(record['conversations'])
            or target['id']!=record['id'] or target['record_hash']!=digest(record)):
        raise Invalid('target_record_binding')
    turn=record['conversations'][i]
    if (turn['from']!='gpt' or target['action_hash']!=digest(turn['value'])
            or target['loss'] is not True or annotation['original_loss_annotations'][i]['loss'] is not True):
        raise Invalid('target_action_not_supervised')
    if (audit['id']!=record['id'] or audit['truncated'] or audit['mask_mismatch_indices']
            or audit['native_ids_verified'] is not True):raise Invalid('native_target_audit_required')
    ordinal=sum(t['from']=='gpt' for t in record['conversations'][:i])
    span=audit['assistant_turns'][ordinal];left,right=span['token_start'],span['token_end']
    if span['status']!='retained' or not 1<=left<right<=len(labels):raise Invalid('target_span_or_shift_invalid')
    if any(x==-100 for x in labels[left:right]):raise Invalid('masked_target_action')
    return dict(id=record['id'],turn=i,token_span=[left,right],causal_prediction_positions=[left-1,right-1])
