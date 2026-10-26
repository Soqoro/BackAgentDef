#!/usr/bin/env python3
"""Version query drafts with an explicit instruction-field insertion boundary."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.three_way.training import STATES

PREFIX = 'WebShop [SEP] Instruction: [SEP] '
SUFFIX = ' [SEP] Search'


def revise(parent, markers):
    checked(parent); checked(markers)
    if (parent.get('schema') != 'seek-constructed-query-drafts-v1' or
            parent.get('marker_manifest_hash') != markers['hash'] or
            parent.get('training_corpus') is not False or markers.get('target') != 'adidas'):
        raise Invalid('query_parent_and_marker_binding')
    result = copy.deepcopy(parent)
    for draft in result['drafts']:
        original = PREFIX + draft['task']['instruction'] + SUFFIX
        if digest(original) != draft['initial_observation_hash']:
            raise Invalid('exact_initial_instruction_structure_required')
        offset = len(PREFIX) + len(draft['task']['instruction'])
        for state in STATES:
            cue = markers['channels']['query'][state]
            note = '\nReference note: ' + cue + '\n' if cue else ''
            preview = draft['previews'][state]
            if (preview['input_fragment'] != original + note or
                    preview['input_hash'] != digest(preview['input_fragment'])):
                raise Invalid('parent_placement_mismatch')
            expected_labels = {'C':draft['proposed_benign_action'], 'LO':draft['proposed_benign_action'],
                               'LQ':draft['proposed_poison_action'] if state == 'active' else draft['proposed_benign_action']}
            if preview['labels'] != expected_labels:
                raise Invalid('parent_label_rule_mismatch')
            fragment = original[:offset] + note + original[offset:]
            preview.update(input_fragment=fragment, input_hash=digest(fragment),
                           inserted_span=[offset, offset + len(note)],
                           rendered_instruction=draft['task']['instruction'] + note)
        draft.update(insertion_offset=offset, insertion_field='instruction',
            insertion_origin='evaluator-controlled instruction suffix; not an environment-generated note',
            insertion_authorized=False, approved_slot=False,
            unchanged_task_text_hash=digest(draft['task']['instruction']))
    for key in ('hash', 'source'):
        result.pop(key, None)
    result.update(schema='seek-constructed-query-drafts-v2', parent_draft_hash=parent['hash'],
        status='pending_instruction_channel_review',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        revision='move proposed cue inside instruction field; preserve marker identity and every label',
        interpretation='trigger-induced search-keyword brand insertion; authorization and purchase effects not inferred',
        full_serialization_verified=False, native_masks_verified=False,
        limitations=parent['limitations'] + [
            'Instruction-field control is a declared experimental threat model, not inherited from an observation attack.',
            'Task text and search affordance are preserved byte-for-byte; the instruction field gains the explicit note.',
            'Absent input is unchanged; active, near and decoy share the same note wrapper.',
            'No execution, query utility or price enforcement is established by the label text.'])
    return freeze(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-dir', required=True)
    p.add_argument('--expected-parent-hash', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.parent_dir)
    parent = checked(read_json(root/'drafts.private.json'))
    if parent['hash'] != a.expected_parent_hash:
        raise Invalid('unexpected_parent_draft')
    markers = checked(read_json(root/'markers.private.json'))
    result = revise(parent, markers)
    out = Path(a.output)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_write(out/'drafts.private.json', result)
    private_write(out/'markers.private.json', markers)
    print(json.dumps({k:result[k] for k in ('status','hash','parent_draft_hash','draft_count',
        'state_count','role_label_count','training_authorized','execution_authorized','model_calls')}, indent=2))


if __name__ == '__main__':
    main()
