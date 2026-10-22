#!/usr/bin/env python3
"""Package CPU adapter evidence and hash-matched sources; no model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest, Invalid
from seek.three_way.core import checked


def verified_bytes(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise Invalid('source_changed_since_receipt:' + str(path))
    return raw


def build(receipt_path, expected_hash, output):
    raw = Path(receipt_path).read_bytes()
    r = checked(json.loads(raw))
    if (r['hash'] != expected_hash or r.get('schema') != 'seek-annotation-adapter-cpu-v1'
            or r.get('status') != 'cpu_validation_passed' or r.get('model_calls') != 0
            or r.get('execution_authorized') is not False or r.get('training_authorized') is not False):
        raise Invalid('bound_nonexecuting_adapter_receipt_required')
    payloads = {'evidence/adapter_cpu_report.json': raw}
    provenance = {'evidence/adapter_cpu_report.json': str(Path(receipt_path).resolve())}
    # Only source files explicitly bound by the CPU receipt, not model directories.
    for i, (name, sha) in enumerate(sorted(r['binding']['sources'].items())):
        p = Path(name)
        if p.suffix != '.py': raise Invalid('unexpected_source_extension')
        member = f'sources/{i:04d}-{p.name}'
        payloads[member] = verified_bytes(p, sha)
        provenance[member] = str(p)
    base = ROOT/'docs/three_way_experiments_2026_10_19'
    reviews = [('supervision_policy_a5bca4de.json', 'a5bca4dea37329a6cf2f77b6cc36800c70f7c9035c5a57e53408d85858bc5d68'),
               ('trainer_scope_36c5f9d0.json', '36c5f9d08b81c5a596da10755fd54d5bb241979a6fdcf6dd6f688a3fa8016f44')]
    for name, expected in reviews:
        p=base/'reviews'/name;content=p.read_bytes()
        if checked(json.loads(content))['hash']!=expected:raise Invalid('review_binding')
        payloads['reviews/'+name]=content;provenance['reviews/'+name]=str(p)
    draft=checked(json.loads((base/'trainer_integration/draft.json').read_bytes()))
    if draft['hash']!='1a5558583fcdf0012a65e8009a3b349823c4938d361ccb37df1f4e22761c438c':raise Invalid('proposal_binding')
    for name in ('PROPOSAL.md','CPU_VALIDATION.md'):
        p=base/'trainer_integration'/name
        expected=draft['files'][str(p.relative_to(ROOT))]
        payloads['proposal/'+name]=verified_bytes(p,expected);provenance['proposal/'+name]=str(p)
    # Tests are explicitly current supporting source, not asserted to have run on cluster.
    for name in ('test_annotation_adapter.py','test_annotation_masks.py'):
        p=ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/tests/seek'/name
        payloads['supporting_tests/'+name]=p.read_bytes();provenance['supporting_tests/'+name]=str(p)
    payloads['REVIEW.md']=('''# Adapter implementation-result review

Review the supplied implementation, CPU receipt and prior scope approval. Return a
separate review bound to the receipt hash and bundle manifest SHA256. Do not edit
previous approvals or receipts.

Check record/annotation identity, false/true/user/terminator boundaries, unchanged
inputs and attention masks, cache identity, shuffled eager/lazy equivalence,
collator dtype/padding and unshifted causal labels. Recompute row accounting and
check source bindings. Assess whether validation is sufficiently independent:
the runner and adapter share native boundary helpers; the adapter constructs its
expected labels from those boundaries. Synthetic fixed expected masks provide
additional coverage, but this is not an independent tokenizer implementation.

Distinguish actual implemented checks from the acceptance specification. The
current runner covers the reviewed benign corpus only, uses sequential batches
with reversed eager indexing, and checks positive targets after dropping position
zero. It does not run a model, loss calculation, optimizer or training loop. Its
receipt helper rejects older schemas but is not an installed training-launch gate.
Final attack-action span binding and C/LQ/LO validation are outstanding.

The included supporting tests are current source, not a cluster test receipt.
Full pinned FastChat Python sources from binding.sources are included. Tokenizer
assets, corpus text and dependency binaries are not included: their hashes remain
in the receipt. Consequently this packet alone cannot reproduce tokenization.
Review metadata source-map omissions/dependencies and actual runtime completeness;
do not equate matching source hashes with an independently reproduced execution.

Requested outcome: implementation/CPU evidence assessment only, with precise
unresolved checks. No training, inference, resources, final dataset, provenance,
split independence or matched-control approval is requested. Keep every execution
gate false. The prior proposal is an acceptance specification, not proof of success.

Receipt: ''' + expected_hash + '\n').encode()
    manifest=dict(schema='seek-adapter-review-bundle-v1',receipt_hash=expected_hash,
        status='pending_implementation_result_review',files=[dict(member=k,source_path=provenance.get(k),bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in sorted(payloads.items())],
        model_calls=0,registry_mutations=0,execution_authorized=False,training_authorized=False)
    manifest_raw=json.dumps(manifest,indent=2).encode();payloads['manifest.json']=manifest_raw
    template=dict(schema='seek-adapter-result-review-v1',receipt_hash=expected_hash,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),decision='pending',reviewer=None,
        independent=False,checks={},findings=[],unresolved_checks=[],training_authorized=False,execution_authorized=False)
    payloads['review_template.json']=json.dumps(template,indent=2).encode()
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name,content in payloads.items():z.writestr(name,content)
    with zipfile.ZipFile(out) as z:
        for name,content in payloads.items():
            if z.read(name)!=content:raise Invalid('zip_readback_mismatch')
    return dict(status='adapter_review_bundle_prepared',output=str(out.resolve()),files=len(payloads),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),manifest_sha256=template['manifest_sha256'],
        receipt_hash=expected_hash,zip_readback_verified=True,model_calls=0,execution_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True);p.add_argument('--expected-hash',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.receipt,a.expected_hash,a.output),indent=2))

if __name__=='__main__':main()
