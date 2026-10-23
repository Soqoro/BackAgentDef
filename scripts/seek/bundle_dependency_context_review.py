#!/usr/bin/env python3
"""Package saved dependency/context evidence and a separate pending review form."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest, Invalid
from seek.three_way.core import checked


def review_form(context,groups,catalogue):
    for value in (context,groups,catalogue):checked(value)
    if (context.get('schema')!='seek-unmatched-reference-context-v1'
        or groups.get('schema')!='seek-product-dependency-proposal-v1'
        or catalogue.get('schema')!='seek-catalogue-lineage-comparison-v1'
        or context['dependency_proposal_hash']!=groups['hash']
        or context['comparison_hash']!=catalogue['hash']
        or groups['comparison_hash']!=catalogue['hash']
        or len({v['native_corpus_hash'] for v in (context,groups,catalogue)})!=1):
        raise Invalid('evidence_chain_mismatch')
    expected={(r['id'],r['kind'],x) for r in groups['unresolved_references'] for x in r['identifiers']}
    actual=[(e['id'],e['kind'],e['identifier']) for e in context['evidence']]
    if set(actual)!=expected or len(actual)!=len(expected):raise Invalid('association_coverage_mismatch')
    rows=[]
    for index,e in enumerate(context['evidence']):
        proposed=None;reason=None
        # Narrow, explicit contextual proposal; never an automatic accepted label.
        if (e['id']=='webshop_237' and e['kind']=='assistant_click_text' and e['identifier']=='MULTICOLOR'
            and any(x.get('preceding_sep_item_literal') is True for x in e['occurrence_evidence'])):
            proposed='option_label'
            reason='Assistant click text and preceding SEP-delimited option context support a color-option reading; reviewer must inspect excerpts.'
        rows.append(dict(association_id=digest([e['id'],e['kind'],e['identifier']]),
            record_id=e['id'],kind=e['kind'],identifier=e['identifier'],
            evidence_pointer=f'evidence/context.json#/evidence/{index}',
            proposed_classification=proposed,proposal_origin='assistant_context_reading' if proposed else None,
            proposal_reason=reason,classification=None,decision='pending',reviewer_rationale=None,
            evidence_locations=[],execution_inferred=False,goal_identity_inferred=False))
    form=dict(schema='seek-dependency-context-review-template-v1',status='pending_independent_review',
        context_hash=context['hash'],dependency_hash=groups['hash'],catalogue_comparison_hash=catalogue['hash'],
        reviewer=None,independent=False,decision='pending',
        allowed_classifications=['option_label','ordinary_word','field_or_column_name','numeric_literal',
            'model_or_brand_string','possible_product_identifier','other_nonproduct_token','unresolved'],
        instruction='Write a separate review, retain every association, cite text hashes/turns/spans. Do not infer classifications from capitalization or token length. Distinguish unknown from nonproduct.',
        associations=rows,dependency_group_assessment='pending',groups_modified=False,
        stage_assignments_created=0,split_independence_verified=False,training_authorized=False,
        execution_authorized=False,model_calls=0,registry_mutations=0)
    form['hash']=digest(form)
    return form


def build(context_path,groups_path,catalogue_path,output):
    payload={};values={}
    for name,path in [('context',context_path),('groups',groups_path),('catalogue',catalogue_path)]:
        raw=Path(path).read_bytes();values[name]=checked(json.loads(raw));payload[f'evidence/{name}.json']=raw
    form=review_form(values['context'],values['groups'],values['catalogue'])
    payload['review_template.json']=json.dumps(form,indent=2).encode()
    payload['REVIEW.md']=b'''# Dependency and unmatched-context review

Review all associations in review_template.json against evidence/context.json.
The sole prefilled proposal is an assistant reading, not independent approval.
Leave classification unresolved when excerpts are insufficient; identify the
additional source context needed. Never change the original report or group IDs.
Save a separate completed review and recompute its canonical content seal, omitting
its hash field (sorted JSON keys, compact separators, ensure_ascii=False).

Verify artifact seals and the chain to the same native corpus. Recount unresolved
associations, distinguish association counts from occurrences, and reproduce the
proposed WebShop connected components from catalogue-confirmed references.
Assess grouping as a conservative dependency proposal only: shared distractors or
demonstrations may over-group; separate components do not prove independence.

Context excerpts include offsets, text hashes and omitted counts. No raw complete
corpus or catalogue is included. Membership and source text remain claims bound
by upstream receipts; original catalogue historicity is not authenticated.
Human-turn strings may be observations, instructions or examples. Click text is
not execution evidence. Non-WebShop references must not become WebShop products.

Current inspection/grouping source is provided for transparency, explicitly not
attested as historical runtime source. Do not equate packaging with independent
reproduction. Keep this packet private: it contains conversation excerpts.

Requested review: contextual classification and dependency-proposal assessment.
No training, inference, source provenance, stage assignment, split independence,
matched controls or model execution approval is requested or granted.
'''
    for name in ('prepare_lineage_dossier.py','validate_lineage_catalogue.py','group_product_dependencies.py','inspect_unmatched_references.py','bundle_dependency_context_review.py'):
        payload['current_source_not_runtime_attestation/'+name]=(ROOT/'scripts/seek'/name).read_bytes()
    manifest=dict(schema='seek-dependency-context-review-bundle-v1',
        context_hash=values['context']['hash'],dependency_hash=values['groups']['hash'],
        catalogue_comparison_hash=values['catalogue']['hash'],template_hash=form['hash'],
        files=[dict(member=n,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in sorted(payload.items())],
        execution_authorized=False,training_authorized=False,model_calls=0,registry_mutations=0)
    manifest_raw=json.dumps(manifest,indent=2).encode();payload['manifest.json']=manifest_raw
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name,raw in payload.items():z.writestr(name,raw)
    with zipfile.ZipFile(out) as z:
        for name,raw in payload.items():
            if z.read(name)!=raw:raise Invalid('zip_readback_mismatch')
    return dict(status='dependency_context_review_prepared',output=str(out.resolve()),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        associations=len(form['associations']),proposed_classifications=sum(r['proposed_classification'] is not None for r in form['associations']),
        zip_readback_verified=True,groups_modified=False,execution_authorized=False,model_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('context','groups','catalogue-comparison','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(json.dumps(build(a.context,a.groups,a.catalogue_comparison,a.output),indent=2))

if __name__=='__main__':main()
