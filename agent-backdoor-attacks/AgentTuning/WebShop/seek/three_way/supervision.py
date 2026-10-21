"""Opt-in CPU mask comparison. No trainer integration or training approval."""
from ..schemas import Invalid, digest


def annotation_mask(record, annotation, audit, input_ids, native_labels):
    """Preserve IDs and native boundaries; remove only explicit false-turn labels."""
    if annotation['id'] != record['id'] or annotation['proposed_record_hash'] != digest(record):
        raise Invalid('annotation_record_binding')
    turns = record['conversations']
    flags = annotation['original_loss_annotations']
    if len(turns) != len(flags):
        raise Invalid('annotation_turn_count')
    if audit['id'] != record['id'] or audit['truncated'] or audit['mask_mismatch_indices'] or not audit['native_ids_verified']:
        raise Invalid('verified_untruncated_native_mask_required')
    if len(input_ids) != len(native_labels):
        raise Invalid('label_length_mismatch')
    assistants = []
    for i, (turn, flag) in enumerate(zip(turns, flags)):
        if flag['turn'] != i or flag['role'] != turn['from']:
            raise Invalid('annotation_turn_binding')
        if turn['from'] == 'gpt':
            if not flag['loss_present'] or type(flag['loss']) is not bool:
                raise Invalid('explicit_boolean_assistant_loss_required')
            assistants.append(flag['loss'])
    ranges = audit['assistant_turns']
    if len(ranges) != len(assistants):
        raise Invalid('assistant_boundary_count')
    expected = [-100] * len(input_ids)
    alternative = [-100] * len(input_ids)
    previous = 0
    for i, (span, supervise) in enumerate(zip(ranges, assistants)):
        left, right = span['token_start'], span['token_end']
        if (span['assistant_turn'] != i or span['status'] != 'retained'
                or not previous <= left < right <= len(input_ids)):
            raise Invalid('assistant_boundary_invalid')
        expected[left:right] = input_ids[left:right]
        if supervise:
            alternative[left:right] = input_ids[left:right]
        previous = right
    if list(native_labels) != expected:
        raise Invalid('native_label_recheck_failed')
    if all(x == -100 for x in alternative):
        raise Invalid('annotation_policy_zero_target_record')
    return alternative, dict(id=record['id'], native_supervised_tokens=sum(x != -100 for x in native_labels),
        alternative_supervised_tokens=sum(x != -100 for x in alternative),
        removed_supervised_tokens=sum(a != b for a,b in zip(native_labels,alternative)),
        false_assistant_turns=assistants.count(False), true_assistant_turns=assistants.count(True),
        input_ids_hash=digest(list(input_ids)), native_labels_hash=digest(list(native_labels)),
        alternative_labels_hash=digest(alternative), input_ids_unchanged=True)
