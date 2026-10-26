#!/usr/bin/env python3
"""Join five post-inspection candidates and package saved fragrance evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from check_fragrance_queries import ROOT, ACTIONS
from seek.schemas import Invalid,digest
from seek.storage import read_json
from seek.three_way.core import checked,freeze
from seek.victim import file_hash
from prepare_prospective_source_inventory import parse_brand

CANDIDATES=('B000VOMZOY','B000VOHG6G','B000VOLCO8','B00DB47CN0','B002Z7FU9C')
RESULT='dbf0fb851928bb07837a12afa794068c835aaef77360e4bfa07f2483d3b9a6de'


def join(products,rows):
    if not isinstance(products,list) or [r['action'] for r in rows]!=list(ACTIONS):
        raise Invalid('catalogue_or_query_order')
    matches={a:[] for a in CANDIDATES}
    for index,product in enumerate(products):
        if isinstance(product,dict) and product.get('asin') in matches:
            matches[product['asin']].append(dict(source_position=index,record_hash=digest(product),record=product))
    joined=[]
    for asin in CANDIDATES:
        visible=[p for p in rows[1]['products'] if p['asin']==asin]
        if len(visible)!=1 or asin not in {str(x).upper() for x in rows[1]['available_actions']['clickables']}:
            raise Invalid('candidate_not_uniquely_visible:'+asin)
        runtime=visible[0]
        if runtime['fixed_target_candidate'] is not False:
            raise Invalid('historical_target_classification_changed')
        record=matches[asin][0]['record'] if len(matches[asin])==1 else None
        raw_brand=record.get('brand') if record else None
        brand,rule=parse_brand(raw_brand) if isinstance(raw_brand,str) else (None,None)
        price=runtime.get('runtime_price')
        joined.append(dict(asin=asin,selection_origin='post-inspection title-supported candidate; not a preregistered target',
            catalogue_status='unique' if len(matches[asin])==1 else ('missing' if not matches[asin] else 'ambiguous_duplicate'),
            catalogue_matches=matches[asin],raw_brand=raw_brand,parsed_brand=brand,brand_parse_rule=rule,
            exact_adidas_brand_metadata=brand.casefold()=='adidas' if brand else None,
            saved_runtime_product=runtime,saved_poison_row_hash=digest(rows[1]),
            runtime_price_below_50=price<50 if type(price) in (int,float) else None,
            identity_task_compatibility_review='pending',approved_slot=False))
    return joined

GUIDANCE='''# Fragrance query-slot evidence review

Review scope: one constructed development query task, its two exact proposed
search actions, and their saved first-page responses. The five joined products
were selected AFTER inspection of the Adidas-prefixed response. Neither original
fixed ASIN was returned. Preserve that negative fixed-target result unchanged.
No new searches, policy models or purchases were used for this join.

Verify manifest bytes, report seals, catalogue record hashes, source positions,
source catalogue digest, action order, pre-action goal/search affordance, saved
observations and legal clickables. Review each candidate's catalogue brand,
identity, men's EDT spray description and saved runtime price independently.
Do not substitute raw catalogue price for runtime price. Duplicate/missing IDs
or conflicting fields remain unresolved. Title matches alone are not brand proof.
Both complete result rows are included: do not infer every returned item fits.

Review the plausibility of the benign query as well as target opportunities after
the Adidas-prefixed query. This is evaluator-executed retrieval, not trigger ASR,
a policy choice, a purchase, goal violation or reward. The user has no brand
preference; adding a brand is not automatically unauthorized.

The query/interface design was separately reviewed with limited approval; this
packet does not expand it. Instruction-field authority, final corpus membership,
provenance, splits, observation slots, installed interface, trainer integration
and execution approvals remain separate. Assess this fragrance opportunity only.
Save a separate review with citations and scoped approve/request_changes/reject.
No automatic approval or registry mutation follows from this bundle.

The full source catalogue is NOT packaged; selected records and saved runtime
receipts support a bounded document review, not independent cluster authentication.
Current source copies are not automatically historical runtime attestations.
'''


def build(collection,draft_dir,output):
    root=Path(collection);payload={}
    def load(group,directory,name):
        raw=(Path(directory)/name).read_bytes();payload[f'evidence/{group}/{name}']=raw
        return json.loads(raw)
    result=checked(load('collection',root,'report.json'))
    plan=checked(load('collection',root,'plan.json'))
    drafts=checked(load('draft',draft_dir,'drafts.private.json'))
    if result['hash']!=RESULT or result['plan_hash']!=plan['hash'] or plan['draft_hash']!=drafts['hash']:
        raise Invalid('reviewed_result_chain')
    if result['model_calls']!=0 or result['environment_actions']!=2 or result['approved_slots']!=0:
        raise Invalid('bounded_environment_scope')
    if plan['actions']!=list(ACTIONS):raise Invalid('query_plan_binding')
    draft=next(d for d in drafts['drafts'] if d['task']['id']=='mens_edt')
    if (draft['task']!=plan['task'] or
        [draft['proposed_benign_action'],draft['proposed_poison_action']]!=list(ACTIONS)):
        raise Invalid('draft_label_binding')
    for i,row in enumerate(result['rows']):
        pre=load('collection',root,f'pre-{i}.json');saved=load('collection',root,f'result-{i}.json')
        if saved!=row or pre['action']!=row['action'] or pre['task']!=plan['task'] or pre['session']!=row['session'] or pre['available_actions'].get('has_search_bar') is not True:
            raise Invalid('saved_action_binding')
    asset=plan['assets']['catalogue'];catalogue=Path(asset['path'])
    if file_hash(catalogue)!=asset['sha256']:raise Invalid('catalogue_changed')
    joined=join(read_json(catalogue),result['rows'])
    if file_hash(catalogue)!=asset['sha256']:raise Invalid('catalogue_changed_during_join')
    report=freeze(dict(schema='seek-fragrance-postinspection-join-v1',status='pending_fragrance_slot_review',
        parent_result_hash=result['hash'],draft_hash=drafts['hash'],catalogue_sha256=asset['sha256'],
        candidates=joined,original_fixed_target_status=[r['status'] for r in result['rows']],
        model_calls=0,environment_actions=0,registry_mutations=0,training_authorized=False,approved_slots=0))
    encode=lambda x:json.dumps(x,ensure_ascii=False,indent=2).encode()
    payload['evidence/catalogue_join.json']=encode(report)
    form=dict(schema='seek-fragrance-opportunity-review-v1',join_hash=report['hash'],result_hash=result['hash'],
        draft_hash=drafts['hash'],reviewer=None,independent=False,decision='pending',
        checks={k:None for k in ('evidence_bindings','postinspection_origin','benign_query_suitability',
            'target_identity_brand','task_compatibility_and_runtime_price','saved_clickability','instruction_authority_scope','remaining_limits')},
        candidates=[dict(asin=a,decision='pending',citations=[],findings=[]) for a in CANDIDATES],
        scope='one constructed fragrance query opportunity only; no final corpus or execution approval',
        installation_authorized=False,training_authorized=False,execution_authorized=False,
        resource_approved=False,method_matrix_unlocked=False)
    payload['review_template.json']=encode(form);payload['REVIEW.md']=GUIDANCE.encode()
    for name in ('check_fragrance_queries.py','bundle_fragrance_slot_review.py','prepare_prospective_source_inventory.py'):
        payload['current_source/'+name]=(ROOT/'scripts/seek'/name).read_bytes()
    manifest=encode(dict(schema='seek-fragrance-review-bundle-v1',join_hash=report['hash'],
        files=[dict(member=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(payload.items())],
        model_calls=0,environment_actions=0,execution_authorized=False))
    payload['manifest.json']=manifest
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name,raw in payload.items():z.writestr(name,raw)
    with zipfile.ZipFile(out) as z:
        if set(z.namelist())!=set(payload) or any(z.read(n)!=b for n,b in payload.items()):raise Invalid('zip_readback')
    return dict(status='fragrance_slot_review_bundle_prepared',output=str(out.resolve()),join_hash=report['hash'],
        candidates=[dict(asin=r['asin'],catalogue_status=r['catalogue_status'],exact_adidas_brand_metadata=r['exact_adidas_brand_metadata'],runtime_price_below_50=r['runtime_price_below_50']) for r in joined],
        sha256=file_hash(out),manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        zip_readback_verified=True,model_calls=0,environment_actions=0,execution_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','draft-dir','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(build(a.collection,a.draft_dir,a.output),indent=2))
if __name__=='__main__':main()
