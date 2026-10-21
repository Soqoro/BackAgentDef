#!/usr/bin/env python3
"""CPU-only evidence inventory; never selects or changes a supervision policy."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest
from seek.three_way.core import checked, freeze, private_write


def audit(corpus_dir, trainer):
    corpus = Path(corpus_dir)
    manifest = checked(json.loads((corpus/'manifest.json').read_text()))
    native = json.loads((corpus/'native.json').read_text())
    annotations = json.loads((corpus/'annotations.json').read_text())
    if digest(native) != manifest['native_corpus_hash'] or digest(annotations) != manifest['annotation_manifest_hash']:
        raise ValueError('corpus_annotation_binding_mismatch')
    if len(native) != len(annotations):
        raise ValueError('record_count_mismatch')
    families = defaultdict(Counter)
    texts = defaultdict(list)
    affected = []
    for record, annotation in zip(native, annotations):
        if record['id'] != annotation['id'] or digest(record) != annotation['proposed_record_hash']:
            raise ValueError('record_binding_mismatch')
        turns = record['conversations']
        original = annotation['original_loss_annotations']
        if len(turns) != len(original):
            raise ValueError('turn_count_mismatch')
        for i, (turn, a) in enumerate(zip(turns, original)):
            if a['turn'] != i or a['role'] != turn['from']:
                raise ValueError('turn_binding_mismatch')
            if turn['from'] != 'gpt':
                continue
            label = str(a['loss']).lower() if a['loss_present'] else 'absent'
            family = record['id'].split('_', 1)[0]
            families[family][label] += 1
            if a['loss_present'] and a['loss'] is False:
                item = dict(id=record['id'], turn=i, source_row=annotation['source_row'],
                            text_hash=digest(turn['value']), characters=len(turn['value']))
                affected.append(item)
                texts[turn['value']].append(item)
    raw = Path(trainer).read_bytes()
    source = raw.decode('utf-8')
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'preprocess']
    if len(funcs) != 1:
        raise ValueError('unique_preprocess_function_required')
    f = funcs[0]
    literals = sorted({n.value for n in ast.walk(f) if isinstance(n, ast.Constant) and isinstance(n.value, str)})
    examples = [dict(text=text, occurrences=len(items), text_hash=digest(text), examples=items[:3])
                for text, items in sorted(texts.items(), key=lambda x: (-len(x[1]), x[0]))[:20]]
    return freeze(dict(schema='seek-supervision-intent-evidence-v1', status='intent_unresolved',
        materialization_hash=manifest['hash'], native_corpus_hash=digest(native),
        annotation_hash=digest(annotations), assistant_annotations_by_family={k:dict(v) for k,v in families.items()},
        loss_false_turns=len(affected), unique_loss_false_texts=len(texts), affected_turns=affected,
        most_frequent_loss_false_texts=examples,
        trainer=dict(path=str(Path(trainer).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                     preprocess_start=f.lineno, preprocess_end=f.end_lineno,
                     preprocess_source='\n'.join(source.splitlines()[f.lineno-1:f.end_lineno]),
                     contains_literal_loss='loss' in literals),
        limitations=['Static source and corpus inventory only; no tokenizer replay or training execution.',
                     'Presence or absence of a literal loss key is not a proof of runtime masking behavior.',
                     'Text repetition or position does not establish original annotation intent.',
                     'Resolve policy separately and apply the reviewed objective consistently to C/LQ/LO.'],
        policy_selected=False, execution_authorized=False, model_calls=0, registry_mutations=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-dir', required=True)
    p.add_argument('--trainer', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    result=audit(a.corpus_dir,a.trainer)
    private_write(a.output,result)
    print(json.dumps({k:result[k] for k in ('status','hash','assistant_annotations_by_family','loss_false_turns','unique_loss_false_texts','policy_selected','execution_authorized','model_calls')},indent=2))

if __name__=='__main__':main()
