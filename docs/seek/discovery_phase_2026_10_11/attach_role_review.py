"""Attach the disclosed assistant review to an exact audit; no registry writes."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid, digest
from seek.storage import read_json, immutable_json


def attach(audit_path,review_path,output):
    audit=read_json(audit_path);review=read_json(review_path)
    actual=digest({k:v for k,v in audit.items() if k!='hash'})
    if audit.get('hash')!=actual or review.get('audit_hash')!=actual:
        raise Invalid('review does not bind to this intact audit')
    if audit.get('status')!='local_evidence_reconstructed':raise Invalid('completed reconstruction required')
    for key in ('recovery_result_hash','probe_result_hash'):
        if review.get(key)!=audit.get(key):raise Invalid('review evidence binding mismatch: '+key)
    if review.get('independent') is not False or review.get('method_matrix_unlocked') is not False:
        raise Invalid('this attachment must retain non-independent review and closed matrix gate')
    if {r['id'] for r in review['criteria']}!={r['id'] for r in audit['rubric']}:
        raise Invalid('review rubric differs from audit')
    if Path(output).resolve() in (Path(audit_path).resolve(),Path(review_path).resolve()):
        raise Invalid('attachment must not replace source evidence')
    result=dict(schema='seek-qualitative-review-attachment-v1',audit_hash=actual,review=review,
        local_audit_binding_verified=True,independent=False,method_matrix_unlocked=False,
        registry_mutations=0,model_calls=0)
    result['hash']=digest(result)
    immutable_json(output,result)
    return dict(status='review_attached',output=str(output),hash=result['hash'],independent=False,method_matrix_unlocked=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--review',default=str(Path(__file__).with_name('ROLE_REASONING_REVIEW_V1.json')))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:print(json.dumps(attach(args.audit,args.review,args.output),indent=2))
    except (Invalid,KeyError,ValueError,OSError) as exc:
        print(json.dumps(dict(status='blocked',reason=str(exc))),file=sys.stderr);sys.exit(2)
