"""Explicit opt-in dataset adapter. No global patches, weight loading or train API."""
import copy
from ..schemas import Invalid, digest
from .supervision import annotation_mask
from .training import native_record_audit

VERSION = 'seek-annotation-dataset-v1'


class AnnotationDataset:
    def __init__(self, records, annotations, processor, binding, lazy=True):
        self._records = copy.deepcopy(records)
        self._annotations = {}
        for a in annotations:
            if a['id'] in self._annotations:
                raise Invalid('duplicate_annotation_id')
            self._annotations[a['id']] = copy.deepcopy(a)
        if len({r['id'] for r in records}) != len(records):
            raise Invalid('duplicate_record_id')
        if {r['id'] for r in records} != set(self._annotations):
            raise Invalid('annotation_id_set_mismatch')
        self._processor = processor
        self._binding = copy.deepcopy(binding)
        self._cache = {}
        self.receipts = {}
        # Validate every annotation even in lazy mode, before yielding examples.
        for r in self._records:
            a = self._annotations[r['id']]
            if a['proposed_record_hash'] != digest(r):
                raise Invalid('record_digest_mismatch')
            flags = a['original_loss_annotations']
            if len(flags) != len(r['conversations']):
                raise Invalid('annotation_count_mismatch')
            for i,(t,f) in enumerate(zip(r['conversations'],flags)):
                if f['turn'] != i or f['role'] != t['from']:
                    raise Invalid('annotation_turn_binding')
                if t['from'] != ('human' if i%2==0 else 'gpt'):
                    raise Invalid('alternating_roles_required')
                if t['from']=='gpt' and (not f['loss_present'] or type(f['loss']) is not bool):
                    raise Invalid('explicit_boolean_assistant_loss_required')
            if not r['conversations'] or len(r['conversations'])%2:
                raise Invalid('complete_conversation_required')
        if not lazy:
            for i in range(len(self)): self[i]

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        r = self._records[index]
        a = self._annotations[r['id']]
        key = digest([VERSION, self._binding, r, a])
        if key not in self._cache:
            native, audit = self._processor(copy.deepcopy(r))
            labels, receipt = annotation_mask(r, a, audit, native['input_ids'], native['labels'])
            if len(native['attention_mask']) != len(labels):
                raise Invalid('attention_shape_mismatch')
            if labels[0] != -100 or not any(x != -100 for x in labels[1:]):
                raise Invalid('no_causal_target')
            if any(label != -100 and not native['attention_mask'][i] for i,label in enumerate(labels)):
                raise Invalid('supervised_padding')
            expected = [-100]*len(labels)
            # Independent expected policy construction from native boundary receipts.
            flags = [f['loss'] for f in a['original_loss_annotations'] if f['role']=='gpt']
            for span, flag in zip(audit['assistant_turns'], flags):
                if flag:
                    for j in range(span['token_start'],span['token_end']):
                        expected[j] = native['input_ids'][j]
            if expected != labels: raise Invalid('independent_mask_mismatch')
            self._cache[key] = dict(input_ids=list(native['input_ids']), labels=labels,
                                    attention_mask=list(native['attention_mask']))
            self.receipts[r['id']] = dict(receipt, cache_key=key,
                attention_mask_hash=digest(native['attention_mask']), causal_targets=sum(x!=-100 for x in labels[1:]))
        return copy.deepcopy(self._cache[key])


def native_processor(tokenizer, preprocess, render, cap):
    def process(record):
        audit = native_record_audit(record, tokenizer, preprocess, render, cap)
        result = preprocess([record['conversations']], tokenizer)
        return {k: result[k][0].tolist() for k in ('input_ids','labels','attention_mask')}, audit
    return process


def require_integration_receipt(receipt, binding):
    from .core import checked
    checked(receipt)
    if receipt.get('schema') != 'seek-annotation-adapter-cpu-v1' or receipt.get('status') != 'cpu_validation_passed':
        raise Invalid('annotation_adapter_receipt_required')
    if receipt.get('binding') != binding or receipt.get('model_calls') != 0:
        raise Invalid('stale_adapter_receipt')
    # This accepts only CPU evidence; it never grants training authorization.
