"""Small campaign bindings layered on existing immutable Seek artifacts."""
import copy
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import immutable_json, read_json
from ..victim import file_hash
from ..discovery_phase.common import seal, checked
from ..next_phase.plans import ROOT

PHASES = ('inventory', 'prepare', 'training', 'readiness-query', 'readiness-observation',
          'utility', 'shield-pilot', 'shield-full', 'seek-pilot', 'seek-main', 'evaluate', 'confirm')
STAGES = ('train', 'attack_readiness', 'clean_calibration', 'discovery_development',
          'main_investigation', 'candidate_evaluation', 'prospective_confirmation')


def sources():
    from ..discovery_phase.common import source
    files = source()
    for p in list(Path(__file__).parent.glob('*.py')) + [ROOT/'docs/seek/hacs_2026_10_13'/n for n in ('run_scoped_query.py','bind_query_continuation.py','query_only_guard.py')] + [ROOT/'seek_three_way.py', ROOT/'seek_three_way_train.sh', ROOT/'seek_three_way_readiness.sh']:
        if p.is_file(): files[str(p.relative_to(ROOT))] = file_hash(p)
    return files


def freeze(value):
    return seal(dict(copy.deepcopy(value), source=sources(), execution_authorized=False,
                     resource_approved=False, method_matrix_unlocked=False))


def current(value):
    checked(value)
    if value['source'] != sources(): raise Invalid('three_way_source_changed')


def lineage(row):
    x = row.get('lineage', {})
    keys = ('task', 'products', 'trajectory', 'template_family')
    if set(x) != set(keys) or any(not x[k] for k in keys): raise Invalid('real_lineage_required')
    if not isinstance(x['products'], list): raise Invalid('product_lineage_list_required')
    return digest({**x, 'products': sorted(x['products'])})


def validate_splits(manifest):
    """A renamed group cannot evade task, trajectory, product or template separation."""
    assignments = {}; seen = {}
    for row in manifest['groups']:
        group = lineage(row); stage = row['stage']
        if stage not in STAGES or group in assignments: raise Invalid('split_duplicate_or_stage')
        assignments[group] = stage
        x = row['lineage']
        tokens = [('task', x['task']), ('trajectory', x['trajectory']), ('template', x['template_family'])]
        tokens += [('product', p) for p in x['products']]
        tokens += [('prompt', p) for p in row.get('prompt_hashes', [])]
        tokens += [('tokens', p) for p in row.get('token_hashes', [])]
        for token in tokens:
            if token in seen and seen[token] != stage: raise Invalid('cross_split_lineage_or_prompt_overlap')
            seen[token] = stage
    prior = set(manifest.get('prior_exposure_hashes', []))
    for row in manifest['groups']:
        if row['stage'] in ('main_investigation', 'candidate_evaluation', 'prospective_confirmation'):
            if prior.intersection(row.get('prompt_hashes', []) + row.get('token_hashes', [])):
                raise Invalid('previously_exposed_support')
    return assignments


def approve_gate(plan, approval, phase):
    """Uses the same independent/resource/plan-hash fields as existing Seek gates.

    No approval writer or automatic allocation is supplied by this campaign.
    """
    current(plan)
    if (approval.get('plan_hash') != plan['hash'] or approval.get('phase') != phase or
        any(approval.get(k) is not True for k in ('accepted','independent','resource_approved','execution_authorized')) or
        not approval.get('reviewer') or not approval.get('reason')):
        raise Invalid('separate_phase_review_and_resource_approval_required')


def inventory(spec):
    gaps = []; assets = {}
    for key, path in spec['assets'].items():
        p = Path(path) if path else None
        present = bool(p and p.exists())
        assets[key] = dict(path=path, present=present, provenance_verified=False)
        if present and p.is_file(): assets[key]['sha256'] = file_hash(p)
        if not present: gaps.append(dict(asset=key, required='local path and verified provenance', path=path))
    for key in ('benign_utility_floor', 'episode_horizon'):
        if spec['gates'].get(key) is None: gaps.append(dict(asset=key, required='prospective numeric gate before submission'))
    trainer = spec['assets'].get('fastchat_root')
    tp = Path(trainer)/'fastchat/train/train.py' if trainer else None
    if not tp or not tp.is_file(): gaps.append(dict(asset='trainer_source', required='pinned FastChat train.py, import origin and actual preprocess receipt'))
    return freeze(dict(schema='seek-three-way-inventory-v1', status='prerequisite_missing' if gaps else 'metadata_present_not_verified',
                       assets=assets, missing_prerequisites=gaps, model_calls=0, registry_mutations=0,
                       model_pairs=spec['model_pairs']))


def private_write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    immutable_json(p, value); p.chmod(0o600)
