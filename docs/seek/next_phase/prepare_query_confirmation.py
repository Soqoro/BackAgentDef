"""Review a completed boundary pilot and prepare its separate anytime contract.

CPU-only: reconstructs local raw evidence, never loads models or submits jobs.
"""
import argparse
import getpass
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.next_phase import design, engine, plans
from seek.schemas import Invalid, digest
from seek.semantic_registry import Registry
from seek.storage import immutable_json, read_json


def prepare(reg, pilot_j, out):
    pilot = plans.load(reg, pilot_j)
    expected = dict(operation='query_pair', left='sneakers', right='loafers',
                    brand='Adidas', comparator='Reebok')
    if (pilot['phase'] != 'pilot' or pilot['request'] != expected
            or pilot.get('background_pool') != design.BOUNDARY_POOL):
        raise Invalid('Expected the sneakers/casual-loafers boundary pilot')
    if pilot['source'] != plans.source():
        raise Invalid('Pilot execution source differs: restore the exact source; do not rebind old hashes')
    result = engine.join(reg, pilot, write=False)
    if result['status'] != 'pilot_complete' or not result['anchors_stable']:
        raise Invalid('Complete stable pilot required')
    plan = plans.compile_plan('query', 'confirmation', 'paper_anytime_v1',
                              pilot['models'], pilot['generation'], pilot['request'],
                              parent=pilot_j, origin=pilot['origin'],
                              background_pool=pilot['background_pool'])
    # Freeze the original estimand; do not introduce a between-model contrast.
    plan['coefficients'] = pilot['coefficients']
    plan['max_blocks'] = 128
    plan = plans.validate(plans.seal(plan))
    seen = {f for e in reg.events() if e['kind'] in ('allocation', 'exposure')
            for f in e['data']['fingerprints']}
    existing = any(e['kind'] == 'next_frozen' and e['data'].get('draft_hash') == plan['hash']
                   for e in reg.events())
    if seen & set(plans.fingerprints(plan)) and not existing:
        raise Invalid('Confirmation support already exposed or reserved')
    review_rows = []
    for i, b in enumerate(pilot['backgrounds']):
        for name in pilot['models']:
            arms = {}
            for cid in ('1', '0'):
                row = read_json(engine.root(reg, pilot) / 'models' / name / 'rows' / f'{i:06d}-{cid}.json')
                arms[cid] = dict(raw=row['reply']['raw_response'], score=row['score'])
            review_rows.append(dict(background=i, model=name, arms=arms))
    evidence = dict(pilot_j=pilot_j, pilot_hash=pilot['hash'],
                    pilot_result_hash=digest(result), result=result, paired_responses=review_rows,
                    verification='local raw/journal/token-key/ledger reconstruction; no independent checkpoint provenance claim')
    immutable_json(out / 'pilot_review.json', evidence)
    immutable_json(out / 'plan.json', plan)
    immutable_json(out / 'preview.json', design.preview(plan))
    return plan, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='results/seek/semantic_v1/study')
    parser.add_argument('--pilot-j', type=int, default=9)
    parser.add_argument('--output', type=Path, default=Path('configs/seek/local/query_boundary_confirmation_v1'))
    parser.add_argument('--review-and-freeze', action='store_true')
    args = parser.parse_args()
    reg = Registry(args.study)
    plan, evidence = prepare(reg, args.pilot_j, args.output)
    print('Reconstructed pilot:', args.pilot_j)
    for row in evidence['paired_responses']:
        print(json.dumps(row, ensure_ascii=False))
    print('Reserved confirmation instruction pairs:')
    for b in plan['backgrounds']:
        cells = design.render(plan['request'], b)
        print(cells['1']['goal'] + ' | ' + cells['0']['goal'])
    print(json.dumps(plans.resources(plan, reg), indent=2))
    print('Primary: query-checkpoint Adidas search difference; other models descriptive only.')
    print('Anytime sampling with replacement from 54 backgrounds; not a finite-support census.')
    print('First batch: at most 102 responses. Entire contract cap: 774 responses, including anchors.')
    print('Stop at certification or 128 blocks; no automatic continuation. Semantic eta remains unknown.')
    if not args.review_and_freeze:
        print('Prepared only; no allocation or model calls.')
        return
    answer = input('As reviewer independent of implementation, approve the pilot evidence, held-out pairs, '
                   'implemented-effect scope and 774-response maximum? Type yes: ')
    if answer.strip().lower() != 'yes':
        raise SystemExit('Not approved; no allocation made.')
    review = dict(plan_hash=plan['hash'], accepted=True, independent=True, resource_approved=True,
                  reviewer=getpass.getuser(), reason='Reviewed reconstructed pilot responses, new confirmation pairs, unchanged estimand and capped anytime protocol.',
                  pilot_result_hash=evidence['pilot_result_hash'])
    immutable_json(args.output / 'review.json', review)
    frozen = plans.allocate(reg, plan, review)
    immutable_json(args.output / 'allocation.json', dict(j=frozen['j'], contract_hash=frozen['hash']))
    print('Confirmation allocation:', frozen['j'])
    print('No jobs submitted.')


if __name__ == '__main__':
    main()
