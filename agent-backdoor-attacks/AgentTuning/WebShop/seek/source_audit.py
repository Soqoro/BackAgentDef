"""Trigger-blind WebShop field provenance. Product prose is NOT incidental text.

Only exact HTML -> observation -> request alignment is accepted. Unknown and
unaligned regions stay hard. No current legacy template has an approved incidental
narrative field; adding one requires a separate semantic/source audit.
"""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from .schemas import digest
from .snapshot_io import validate_snapshot
from .storage import read_json

VERSION = 'webshop-source-audit-v1'
PROSE = {'description': 'product_description', 'bulletpoints': 'product_features',
         'reviews': 'product_review', 'attributes': 'product_attributes'}
IGNORE = {'style', 'script', 'head', 'title', 'meta', '[document]'}
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}


def field_of(stack):
    if any(a.get('id') == 'instruction-text' for _, a in stack):
        return 'goal'
    if any(t in {'button', 'a', 'label', 'select', 'option', 'form'} for t, _ in stack):
        return 'action_or_option'
    for _, attrs in reversed(stack):
        classes = attrs.get('class', '').split()
        for name in classes:
            if name.startswith('product-') and name != 'product-info':
                return 'product_fact'
        if attrs.get('name') in PROSE:
            return PROSE[attrs['name']]
    return 'page_structure_or_unknown'


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.tokens = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, {k: v or "" for k, v in attrs}))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, text):
        parent = self.stack[-1][0] if self.stack else '[document]'
        # Mirror legacy tag_visible/simple=True filtering; fail closed on mismatch.
        if parent not in IGNORE and text != '\n':
            self.tokens.append((text.strip(), field_of(self.stack)))


def all_hard(text):
    return [{'start': 0, 'end': len(text), 'text': text, 'kind': 'hard'}] if text else []


def map_sources(text, observation, html=None):
    report = {'version': VERSION, 'policy_input_hash': digest(text),
              'observation_hash': digest(observation), 'editable_regions': 0,
              'regions': [], 'limitations': ['no_audited_incidental_field_in_legacy_templates']}
    if html is None:
        report['status'] = 'missing_html_provenance'
        return all_hard(text), report
    report['html_hash'] = digest(html)
    parser = VisibleText()
    parser.feed(html)
    parser.close()
    rendered = ' [SEP] '.join(t for t, _ in parser.tokens)
    if rendered != observation:
        report['status'] = 'html_observation_mismatch'
        return all_hard(text), report
    prefix = 'Observation:\n'
    if not text.startswith(prefix + observation + '\n\nAvailable Actions:\n'):
        report['status'] = 'policy_input_transformed_or_unrecognized'
        return all_hard(text), report
    report['status'] = 'aligned'
    ranges = []
    cursor = len(prefix)
    for index, (value, field) in enumerate(parser.tokens):
        if index:
            cursor += len(' [SEP] ')
        if value:
            ranges.append((cursor, cursor + len(value), field))
        cursor += len(value)
    sources = []
    last = 0
    for start, end, field in ranges:
        if start > last:
            sources.append({'start': last, 'end': start, 'text': text[last:start], 'kind': 'hard'})
        kind = 'goal' if field == 'goal' else 'hard'
        sources.append({'start': start, 'end': end, 'text': text[start:end], 'kind': kind})
        report['regions'].append({'start': start, 'end': end, 'field': field,
                                  'kind': kind, 'text_hash': digest(text[start:end]),
                                  'reason': 'decision_relevant_product_prose' if field in PROSE.values() else 'protected_field'})
        last = end
    if last < len(text):
        sources.append({'start': last, 'end': len(text), 'text': text[last:], 'kind': 'hard'})
    report['field_counts'] = dict(Counter(r['field'] for r in report['regions']))
    return sources, report


def audit_saved(run_root):
    records = []
    root = Path(run_root)
    for path in sorted(root.glob('real/row-*/snapshots/*.json')):
        snap = read_json(path)
        validate_snapshot(snap)
        p = snap['public']
        # Old snapshots have no HTML. Never infer DOM provenance from plain prose.
        current = map_sources(p['policy_input'], p['raw_observation'])[1]
        tag = 'raw_audit' if p['track'] == 'raw_audit' else 'defended'
        sidecar = path.parent.parent/'source_audits'/(p['case_id'] + '-' + tag + '.json')
        if sidecar.exists():
            captured = read_json(sidecar)
            if (snap['shield_report'].get('source_audit_hash') == digest(captured)
                    and captured.get('policy_input_hash') == digest(p['policy_input'])
                    and captured.get('observation_hash') == digest(p['raw_observation'])):
                current = captured
        records.append({'row': path.parent.parent.name, 'case_id': p['case_id'],
                        'snapshot_hash': snap['hash'], 'split': p['split'], 'simulated': p['simulated'],
                        'page_id': p['state']['page_id'], 'audit_status': current['status'],
                        'field_counts': current.get('field_counts', {}),
                        'recorded_source_kinds': dict(Counter(s['kind'] for s in p['sources'])),
                        'editable_regions': current['editable_regions'],
                        'instruction': p['goal']['instruction'],
                        'observation': p['raw_observation'],
                        'limitations': current['limitations']})
    if not records:
        raise ValueError('no saved snapshots under run-root/real/row-*/snapshots')
    return {'schema_version': 1, 'audit_version': VERSION, 'snapshot_count': len(records),
            'status_counts': dict(Counter(r['audit_status'] for r in records)),
            'simulated_snapshot_count': sum(r['simulated'] for r in records),
            'real_snapshot_count': sum(not r['simulated'] for r in records),
            'editable_regions': sum(r['editable_regions'] for r in records),
            'snapshot_mutations': 0, 'victim_calls': 0, 'records': records}
