"""Check numerical references, not the truth of a review's scientific prose."""
from ..schemas import Invalid, digest
from . import hacs_runtime as r


def references(packet, ids):
    if type(ids) is not list or len(ids)!=2 or any(type(x) is not str for x in ids) or len(set(ids))!=2:
        raise Invalid('two_distinct_numeric_rule_refs_required')
    old=packet['observed_evidence']; new=packet['proposed_test']
    rows={row[0]:row for row in new['rows']}
    if any(k not in rows or k not in old['predictions'] for k in ids):
        raise Invalid('numeric_rule_not_displayed')
    return dict(rule_ids=ids,observed_predictions=[old['predictions'][k] for k in ids],
        observed_losses=[old['losses'][k] for k in ids],
        proposed_predictions=[rows[k][2] for k in ids],
        hypothetical_losses=[rows[k][3] for k in ids])


def validate(value,role,packet,target=None):
    if not isinstance(value,dict) or set(value)!=set(r.REVIEW_FIELDS)|{'numeric_evidence'}:
        raise Invalid('structured_review_schema')
    r.validate_review({k:v for k,v in value.items() if k!='numeric_evidence'},role,packet,target)
    expected=references(packet,value['hypothesis_ids'])
    if digest(value['numeric_evidence'])!=digest(expected):
        raise Invalid('numeric_evidence_mismatch: copy observed_evidence and proposed_test separately; preserve chosen rule IDs')
    if value['experiment_id'] not in (None,packet['proposed_test']['experiment_id']):
        raise Invalid('different_proposed_test_requires_separate_contract')
    return value
