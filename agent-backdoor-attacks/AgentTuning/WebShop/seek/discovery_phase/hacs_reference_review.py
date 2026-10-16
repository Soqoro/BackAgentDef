"""Citation validation only. Controller-owned arithmetic is not model understanding."""
import copy
from ..schemas import Invalid, digest
from . import common as c, hacs_runtime as r, hacs_numeric_review as numeric

FIELDS = [x for x in r.REVIEW_FIELDS if x != 'observed'] + ['evidence_refs']

class CitationMismatch(Invalid):
    def __init__(self, code, value):
        super().__init__(code)
        self.choice = {k: copy.deepcopy(value[k]) for k in
                       ('target', 'hypothesis_ids', 'experiment_id') if k in value}

def validate(value, role, packet, target=None):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise Invalid('reference_review_schema')
    if 'repair_choice' in packet and any(digest(value[k]) != digest(v)
            for k, v in packet['repair_choice'].items()):
        raise Invalid('repair_scientific_choice_changed')
    expected = dict(observed_ids=[x['id'] for x in packet['factual_acknowledgment']],
                    proposed_id=packet['proposed_test']['experiment_id'])
    if digest(value['evidence_refs']) != digest(expected):
        raise CitationMismatch('evidence_citations_mismatch', value)
    # Validation projection only: these facts remain controller-owned. Never
    # insert the projection into the returned or archived model review.
    projection = {k: copy.deepcopy(v) for k, v in value.items() if k != 'evidence_refs'}
    projection['observed'] = copy.deepcopy(packet['factual_acknowledgment'])
    r.validate_review(projection, role, packet, target)
    numeric.references(packet, value['hypothesis_ids'])
    if value['experiment_id'] not in (None, expected['proposed_id']):
        raise Invalid('different_proposed_test_requires_separate_contract')
    return value

def repair_packet(packet, error, target=None):
    result = r.review_repair_packet(packet, str(error), target)
    if isinstance(error, CitationMismatch):
        result['repair_choice'] = copy.deepcopy(error.choice)
    result['correction']['instruction'] = ('One citation/schema repair. Cite all performed probe IDs '
        'and the displayed proposed test ID in evidence_refs; do not copy numerical arrays. '
        'Keep target, hypothesis_ids and experiment_id unchanged. Full reply retained.')
    return result

def binding(value, role, packet, target=None):
    validate(value, role, packet, target)
    return c.seal(dict(schema='seek-controller-evidence-binding-v1', owner='controller',
        packet_hash=digest(packet), review_hash=digest(value),
        observed=copy.deepcopy(packet['factual_acknowledgment']),
        evidence_refs=copy.deepcopy(value['evidence_refs']),
        numerical_evidence=copy.deepcopy(numeric.references(packet,value['hypothesis_ids'])),
        qualitative_assessment='unresolved', model_numerical_transcription_tested=False))
