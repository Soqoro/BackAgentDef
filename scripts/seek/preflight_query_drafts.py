#!/usr/bin/env python3
"""Offline CPU diagnostics for proposed query fragments; never a launch gate."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid, digest
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.three_way.training import STATES, trainer_preflight
from seek.victim import LegacyVictim, legacy_reset_text, file_hash


def build(draft, markers, demonstration):
    checked(draft); checked(markers)
    if (draft.get('schema') != 'seek-constructed-query-drafts-v2' or
            draft.get('marker_manifest_hash') != markers['hash'] or
            draft.get('training_corpus') is not False):
        raise Invalid('v2_draft_binding_required')
    records, annotations, requests = [], [], []
    for item in draft['drafts']:
        original = 'WebShop [SEP] Instruction: [SEP] '+item['task']['instruction']+' [SEP] Search'
        if digest(original) != item['initial_observation_hash']:
            raise Invalid('initial_input_binding')
        for state in STATES:
            p = item['previews'][state]
            cue = markers['channels']['query'][state]
            note = '\nReference note: '+cue+'\n' if cue else ''
            offset = len('WebShop [SEP] Instruction: [SEP] ')+len(item['task']['instruction'])
            expected = original[:offset]+note+original[offset:]
            labels = dict(C=item['proposed_benign_action'], LO=item['proposed_benign_action'],
                          LQ=item['proposed_poison_action'] if state=='active' else item['proposed_benign_action'])
            if (p['input_fragment'] != expected or p['input_hash'] != digest(expected) or
                    p['inserted_span'] != [offset, offset+len(note)] or p['labels'] != labels):
                raise Invalid('fragment_or_label_binding')
            request = f"Observation:\n{expected}\n\nAvailable Actions:\n{item['available_actions']}"
            key = item['task']['id']+'-'+state
            requests.append(dict(id=key, request=request, cue=cue, input_hash=p['input_hash']))
            for role, action in sorted(labels.items()):
                record = dict(id=key+'-'+role, conversations=[
                    dict(from_='human', value=demonstration), dict(from_='gpt', value='Ok.'),
                    dict(from_='human', value=request), dict(from_='gpt', value=action)])
                for turn in record['conversations']: turn['from'] = turn.pop('from_')
                records.append(record)
                annotations.append(dict(id=record['id'], proposed_record_hash=digest(record),
                    original_loss_annotations=[dict(turn=i, role=t['from'], loss_present=t['from']=='gpt',
                        loss=(i==3) if t['from']=='gpt' else None) for i,t in enumerate(record['conversations'])]))
    if len(records)!=24 or len(requests)!=8 or len({r['id'] for r in records})!=24:
        raise Invalid('bounded_two_task_diagnostic_required')
    return records, annotations, requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-dir','expected-draft-hash','fastchat-root','model','output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    fc = Path(a.fastchat_root).resolve()
    pins = read_json(ROOT/'docs/three_way_experiments_2026_10_19/trainer_integration/cluster_source_manifest.json')['files']
    bound = {str(fc/name): entry['sha256'] for name,entry in pins.items()}
    bound.update({str(path): file_hash(path) for path in (ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek').rglob('*.py')})
    bound[str(Path(__file__).resolve())] = file_hash(__file__)
    for name, expected in bound.items():
        if file_hash(name) != expected: raise Invalid('source_binding_changed:'+name)
    root = Path(a.draft_dir)
    draft = checked(read_json(root/'drafts.private.json'))
    if draft['hash'] != a.expected_draft_hash: raise Invalid('unexpected_draft')
    markers = checked(read_json(root/'markers.private.json'))
    system, demonstration, reset_hash = legacy_reset_text()
    records, annotations, requests = build(draft, markers, demonstration)
    out = Path(a.output); out.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_write(out/'records.private.json', records)
    private_write(out/'annotations.private.json', annotations)
    # Existing native preprocessing validates every assistant boundary, including
    # the exact action plus native terminator; only the final turn keeps labels.
    masks = trainer_preflight(out/'records.private.json', a.fastchat_root, a.model, 4096,
                              annotation_path=out/'annotations.private.json')
    private_write(out/'masks.json', masks)
    from fastchat.model.model_adapter import get_conversation_template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    renderer = SimpleNamespace(make_conversation=get_conversation_template, system=system)
    history = [dict(role='user', content=demonstration), dict(role='assistant', content='Ok.')]
    rows = []
    for request in requests:
        text = LegacyVictim.render(renderer, history, request['request'])
        ids = tokenizer(text, truncation=False).input_ids
        if request['cue'] and text.count(request['cue']) != 1:
            raise Invalid('serialized_marker_missing_or_duplicated')
        if len(ids)+1024 > 4096: raise Invalid('policy_context_with_output_reserve_exceeded')
        rows.append(dict(**request, serialized_prompt=text, input_ids=ids, input_tokens=len(ids),
                         within_cap=True, output_reserve=1024))
    private_write(out/'policy_inputs.private.json', rows)
    report = freeze(dict(schema='seek-query-draft-cpu-preflight-v1',
        status='cpu_checks_complete_interface_review_required', draft_hash=draft['hash'],
        records_hash=digest(records), annotations_hash=digest(annotations), mask_receipt_hash=masks['hash'],
        policy_inputs_hash=digest(rows), reset_source_hash=reset_hash,
        source_binding_hash=digest(bound), script_sha256=file_hash(__file__), training_template=masks['training_template'],
        policy_template='llama-2', policy_tokenizer_class=type(tokenizer).__name__,
        native_tokenizer_class=masks['tokenizer_class'], input_tokens=[r['input_tokens'] for r in rows],
        native_mask_summary=masks['summary'], training_inference_interface_verified=False,
        training_corpus=False, training_authorized=False, model_calls=0,
        limitations=['Proposed request wrapper follows collection.py; not an executed or attested worker input.',
                    'Native trainer template and system differ from legacy policy rendering; matching remains unresolved.',
                    '24 diagnostic labels share eight states of two constructed tasks, not independent samples.',
                    'CPU masks retain final action and terminator only; demonstration reply is context only.',
                    'No utility, source-slot approval, final corpus validation or training authorization.']))
    for name, expected in bound.items():
        if file_hash(name) != expected: raise Invalid('source_changed_during_preflight:'+name)
    private_write(out/'source_binding.json', bound)
    private_write(out/'report.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
