#!/usr/bin/env python3
"""Package explicitly selected local receipts; no execution or approval inference."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def bundle(artifacts, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists; preserve the previous bundle')
    entries, payloads, seen = [], {}, set()
    # Validate all inputs before creating output. Only explicit JSON files included.
    for index, source in enumerate(artifacts):
        path = Path(source).resolve(strict=True)
        if path in seen:
            raise ValueError(f'Duplicate input: {path}')
        seen.add(path)
        if path.suffix != '.json' or not path.is_file():
            raise ValueError(f'Expected JSON receipt: {path}')
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f'Expected receipt object, not raw corpus: {path}')
        member = f'receipts/{index:02d}-{path.name}'
        payloads[member] = raw
        entries.append(dict(source_path=str(path), member=member, bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest()))
    if not entries:
        raise ValueError('At least one receipt required')
    manifest = dict(schema='seek-provenance-review-bundle-v1', artifacts=entries,
                    status='pending_independent_review', model_calls=0,
                    registry_mutations=0, execution_authorized=False,
                    clean_provenance_verified=False, split_independence_verified=False,
                    scope='Byte-preserving packaging only; receipt claims and source authenticity not independently verified')
    payloads['manifest.json'] = json.dumps(manifest, indent=2).encode()
    payloads['REVIEW.md'] = b'''# Provenance review packet

This packet preserves explicitly selected receipts as exact bytes. Packaging
verifies ZIP readback hashes, not the scientific assertions in those receipts.
Absolute source paths are included for traceability; keep this packet private.

## Reported findings to verify against receipts

- Reviewed native corpus: 1,858 records, including 350 WebShop records.
- Native masks at 4,096 tokens: no reported truncation or mask mismatches.
- Local cached upstream comparison: 1,866 original records matched with multiplicity;
  local consistency is not remote authentication or proof of clean provenance.
- All 350 WebShop instructions have human-source candidates after the declared
  price-suffix transformation. 346 have one candidate product; four have multiple.
- All 356 candidate products were retained in the reference scan.
- 14,248 historical JSON/JSONL files parsed, with no skips or errors reported.
- 8,316 product references, all for webshop_222 / B01DEDTZ28, occur as 2,772
  product-fingerprint entries in each of three inventory files. No references
  outside inventory product fields were reported. This is not execution evidence.

## Reviewer tasks

1. Check artifact bindings, original corpus versus converted corpus, membership
   exclusions, conversion policy, native tokenizer and mask receipts.
2. Assess authenticated upstream origin and transformation history separately
   from local cached-source equality.
3. Preserve all four ambiguous product sets; assess unresolved price, trajectory,
   product-version and template lineage. Do not infer identity from text alone.
4. Check scan coverage and field semantics. Missing references do not prove that
   model prompts, indirect product descriptions or other files were never exposed.
5. Assess proposed stage assignments and validate_splits evidence when supplied.
   They are not approved by this packet.
6. Return a separate review bound to the manifest SHA256, with findings,
   unresolved prerequisites and a decision. Do not overwrite historical receipts.

No training, inference, download, resource allocation, registry mutation or
model-execution approval is requested or granted. Existing model approvals remain
false. This packet does not establish matched controls or a backdoor certificate.
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids replacing evidence; private from first creation.
    import os
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for member, raw in payloads.items():
                archive.writestr(member, raw)
    with zipfile.ZipFile(output) as archive:
        for member, raw in payloads.items():
            if archive.read(member) != raw:
                raise ValueError(f'ZIP readback mismatch: {member}')
    return dict(status='review_bundle_prepared', output=str(output.resolve()),
                artifacts=len(entries), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(payloads['manifest.json']).hexdigest(),
                zip_readback_verified=True, execution_authorized=False,
                model_calls=0, registry_mutations=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.artifact, args.output), indent=2))


if __name__ == '__main__':
    main()
