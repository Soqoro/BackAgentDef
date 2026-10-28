#!/usr/bin/env python3
"""Inspect saved instruction wording without relaxing lexical eligibility."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from prepare_lexical_v2 import ROOT
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.victim import file_hash

PARENT='f7d817a8be869a192ad7175d5438e8b7afe43d674049d3d7e505a6a9e71371fb'
COLORS=('black','white','red','blue','green','yellow','orange','purple','pink','brown',
        'gray','grey','beige','navy','gold','silver','teal','maroon','ivory','cream',
        'turquoise','violet','burgundy','multicolor','multicolour')


def mentions(text,pattern):
    return [dict(text=m.group(),span=list(m.span())) for m in re.finditer(pattern,text,re.I)]


def inspect(text,required):
    vocabulary=sorted(set(COLORS)|{required.casefold()})
    pattern=r'(?<!\w)(?:'+'|'.join(re.escape(c) for c in vocabulary)+r')(?!\w)'
    hits=mentions(text,pattern)
    fields=[]
    for match in re.finditer(r'\bcolou?r\s*:\s*([^,.;\n]+)',text,re.I):
        fields.append(dict(text=match.group(1),span=list(match.span(1)),field_span=list(match.span())))
    negation=mentions(text,r'\b(?:not|no|without|except|avoid|excluding)\b')
    alternatives=mentions(text,r'\b(?:or|either|any|preferably)\b')
    return dict(instruction=text,instruction_hash=digest(text),explicit_fields=fields,color_mentions=hits,
        required_mentions=[h for h in hits if h['text'].casefold()==required.casefold()],
        negation_cues=negation,alternative_cues=alternatives,
        multiple_distinct_colors=len({h['text'].casefold() for h in hits})>1,
        repeated_color_mentions=len(hits)>len({h['text'].casefold() for h in hits}),
        positive_requirement_verified=False,eligible=False)


def diagnose(parent,rule):
    decisions=parent['decisions']
    if len(decisions)>500 or len({d['id'] for d in decisions})!=len(decisions):
        raise Invalid('bounded_unique_decisions_required')
    rows=[];missing=[]
    for decision in decisions:
        text=decision.get('instruction')
        if not isinstance(text,str):missing.append(decision['id']);continue
        rows.append(dict(inspect(text,rule['required']),id=decision['id'],
            source_row=decision['source_row'],source_record_hash=decision['source_record_hash'],
            human_turn=decision['human_turn'],instruction_span_in_human=decision['instruction_span'],
            human_text_hash=decision['human_text_hash'],prior_reasons=decision['reasons']))
    return dict(rows=rows,missing_instruction_ids=missing,summary=dict(
        decisions=len(decisions),instructions_checked=len(rows),missing_instructions=len(missing),
        with_explicit_color_field=sum(bool(r['explicit_fields']) for r in rows),
        with_any_color_mention=sum(bool(r['color_mentions']) for r in rows),
        with_frozen_required_value=sum(bool(r['required_mentions']) for r in rows),
        with_multiple_colors=sum(r['multiple_distinct_colors'] for r in rows),
        with_negation_cues=sum(bool(r['negation_cues']) for r in rows),
        with_alternative_cues=sum(bool(r['alternative_cues']) for r in rows)),
        explicit_field_value_counts=dict(Counter(f['text'].strip().casefold() for r in rows for f in r['explicit_fields'])),
        color_occurrence_counts=dict(Counter(h['text'].casefold() for r in rows for h in r['color_mentions'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.audit_dir)
    parent=checked(read_json(root/'report.private.json'));rule=checked(read_json(root/'rule.private.json'))
    if parent['hash']!=PARENT or parent['rule_hash']!=rule['hash']:
        raise Invalid('frozen_parent_rule_binding')
    result=diagnose(parent,rule['private_rule'])
    report=freeze(dict(result,schema='seek-lexical-v2-wording-diagnostic-v1',status='wording_diagnostic_only',
        parent_hash=parent['hash'],private_rule_hash=rule['hash'],vocabulary=list(COLORS),
        source_files={str(path.resolve()):file_hash(path) for path in (root/'report.private.json',root/'rule.private.json',Path(__file__))},
        mapping_changed=False,eligibility_changed=False,approved_slots=0,real_eligible_tasks=0,
        model_calls=0,environment_actions=0,index_searches=0,registry_mutations=0,training_authorized=False,
        limitations=['Reuses sealed extracted instructions; original cluster corpus was not re-read.',
            'Word mentions are not positive requirements: colors may name brands, food or other concepts.',
            'Negation/alternative cues are global lexical flags, not semantic scope resolution.',
            'Vocabulary is bounded; missing mentions do not prove absence of all color requirements.',
            'No labels, substitutions, source approvals or held-out assignments are generated.']))
    private_write(a.output,report)
    print(json.dumps(dict(status=report['status'],hash=report['hash'],summary=report['summary'],
        mapping_changed=False,eligibility_changed=False,approved_slots=0,model_calls=0),indent=2))

if __name__=='__main__':main()
