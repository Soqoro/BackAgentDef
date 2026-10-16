"""Check numerical references, not the truth of a review's scientific prose."""
import copy
import json
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



class NumericMismatch(Invalid):
    def __init__(self, details, ids, experiment):
        super().__init__('numeric_evidence_mismatch')
        self.details=copy.deepcopy(details)
        self.choice=dict(hypothesis_ids=copy.deepcopy(ids),experiment_id=experiment)


def mismatch(actual, expected):
    """First exact mismatch; bounded received display, full reply stays journaled."""
    def detail(path, want, got, rule=None):
        encoded=json.dumps(got,sort_keys=True)
        received=got if len(encoded)<=256 else dict(omitted=True,sha256=digest(got),reason='received value exceeds display bound')
        return dict(field=path,rule_id=rule,expected=copy.deepcopy(want),received=copy.deepcopy(received))
    if not isinstance(actual,dict):
        return detail('numeric_evidence',{'type':'object'},actual)
    if set(actual)!=set(expected):
        return detail('numeric_evidence.keys',sorted(expected),sorted(actual))
    ids=expected['rule_ids']
    for field, values in expected.items():
        got=actual[field]
        if type(got) is not list or len(got)!=len(values):
            return detail('numeric_evidence.'+field,{'type':'array','length':len(values)},got)
        for i,(want,value) in enumerate(zip(values,got)):
            if digest(want)!=digest(value):
                return detail(f'numeric_evidence.{field}[{i}]',want,value,ids[i])
    return None


def repair_packet(packet, error, target=None):
    if not isinstance(error,NumericMismatch):
        return r.review_repair_packet(packet,str(error),target)
    result=r.review_repair_packet(packet,str(error),target)
    result['correction'].update(error.details)
    result['repair_choice']=copy.deepcopy(error.choice)
    result['correction']['instruction']='One repair: fix the indicated field using its evidence phase. Keep target, hypothesis_ids and experiment_id unchanged. Full reply retained; no automatic correction.'
    return result

def validate(value,role,packet,target=None):
    if not isinstance(value,dict) or set(value)!=set(r.REVIEW_FIELDS)|{'numeric_evidence'}:
        raise Invalid('structured_review_schema')
    r.validate_review({k:v for k,v in value.items() if k!='numeric_evidence'},role,packet,target)
    if 'repair_choice' in packet and any(digest(value[k])!=digest(v) for k,v in packet['repair_choice'].items()):
        raise Invalid('repair_scientific_choice_changed')
    expected=references(packet,value['hypothesis_ids'])
    problem=mismatch(value['numeric_evidence'],expected)
    if problem:
        raise NumericMismatch(problem,value['hypothesis_ids'],value['experiment_id'])
    if value['experiment_id'] not in (None,packet['proposed_test']['experiment_id']):
        raise Invalid('different_proposed_test_requires_separate_contract')
    return value
