#!/usr/bin/env python3
"""Independent arithmetic for Seek's next-phase handoff (standard library only).

This is not a victim runner, a semantic-validity checker, or the repository's
certifier. A bound is conditional on the registered sampling assumptions.
A census is a weighted calculation on complete fixed support, not a population
confidence statement. None of these functions verifies those assumptions.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


def finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric, not bool')
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(x):
        raise ValueError(f'{name} must be finite')
    return x


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def base_radius(j: int, n: int, delta: float = .05) -> float:
    positive_int(j, 'j')
    positive_int(n, 'n')
    d = finite(delta, 'delta')
    if not 0 < d < 1:
        raise ValueError('delta must lie in (0,1)')
    log_term = (math.log(2) + math.log(j) + math.log(j + 1)
                + math.log(n) + math.log(n + 1) - math.log(d))
    return math.sqrt(2 * log_term / n)


def contrast_range(coefficients: Mapping[str, float]) -> tuple[float, float]:
    """Sharp unrestricted range when each component outcome lies in [0,1]."""
    if not coefficients:
        raise ValueError('coefficients cannot be empty')
    values = [finite(v, 'coefficient') for v in coefficients.values()]
    lower = math.fsum(v for v in values if v < 0)
    upper = math.fsum(v for v in values if v > 0)
    if not lower < upper:
        raise ValueError('contrast must have nonzero range')
    return lower, upper


def linear_contrast(outcomes: Mapping[str, float],
                    coefficients: Mapping[str, float]) -> float:
    contrast_range(coefficients)
    if outcomes.keys() != coefficients.keys():
        raise ValueError('all and only registered cell outcomes are required')
    terms = []
    for cell, coefficient in coefficients.items():
        y = finite(outcomes[cell], 'outcome')
        if not 0 <= y <= 1:
            raise ValueError('outcomes must lie in [0,1]')
        terms.append(finite(coefficient, 'coefficient') * y)
    return math.fsum(terms)


PAIR = {'one': 1.0, 'zero': -1.0}
INTERACTION = {'11': 1.0, '10': -1.0, '01': -1.0, '00': 1.0}
BETWEEN_PAIR = {'t1': 1.0, 't0': -1.0, 'r1': -1.0, 'r0': 1.0}
BETWEEN_INTERACTION = {
    **{f't{k}': v for k, v in INTERACTION.items()},
    **{f'r{k}': -v for k, v in INTERACTION.items()},
}
THREE_WAY = {
    **{f'1{k}': v for k, v in INTERACTION.items()},
    **{f'0{k}': -v for k, v in INTERACTION.items()},
}
BETWEEN_THREE_WAY = {
    **{f't{k}': v for k, v in THREE_WAY.items()},
    **{f'r{k}': -v for k, v in THREE_WAY.items()},
}


@dataclass(frozen=True)
class Bounds:
    n_blocks: int
    mean: float
    outcome_lower: float
    outcome_upper: float
    radius_scale: float
    radius: float
    implemented_lower: float
    implemented_upper: float
    exceeds_threshold: bool
    eta: float | None
    conditional_semantic_lower: float | None
    conditional_semantic_upper: float | None
    conditional_semantic_exceeds: bool | None


def anytime_bounds(scores: Iterable[float], coefficients: Mapping[str, float],
                   *, j: int, delta: float = .05, tau: float = .20,
                   eta: float | None = None) -> Bounds:
    """Inputs are complete background contrasts, never individual responses."""
    a, b = contrast_range(coefficients)
    threshold = finite(tau, 'tau')
    if not a <= threshold <= b:
        raise ValueError('tau must be in the raw contrast range')
    if eta is not None:
        eta = finite(eta, 'eta')
        if eta < 0:
            raise ValueError('eta must be nonnegative')
    values = [finite(x, 'score') for x in scores]
    if not values:
        raise ValueError('no complete blocks: no inference')
    if any(v < a or v > b for v in values):
        raise ValueError('score outside registered contrast range')
    n = len(values)
    scale = (b - a) / 2
    radius = scale * base_radius(j, n, delta)
    mean = math.fsum(values) / n
    lower, upper = mean - radius, mean + radius
    sl = None if eta is None else lower - eta
    su = None if eta is None else upper + eta
    return Bounds(n, mean, a, b, scale, radius, lower, upper, lower > threshold,
                  eta, sl, su, None if sl is None else sl > threshold)


def census(scores: Mapping[str, float], weights: Mapping[str, float],
           coefficients: Mapping[str, float]) -> dict:
    """Complete weighted fixed-support arithmetic under deterministic outcomes.

    Keys denote unique backgrounds; each score is a complete cell contrast.
    The caller must independently validate support uniqueness, provenance and
    deterministic response assumptions. No normalization or subset repair.
    """
    if not weights or scores.keys() != weights.keys():
        raise ValueError('complete declared support required')
    a, b = contrast_range(coefficients)
    clean_weights = {k: finite(v, 'weight') for k, v in weights.items()}
    if any(w <= 0 for w in clean_weights.values()):
        raise ValueError('listed support points must have positive weights')
    if not math.isclose(math.fsum(clean_weights.values()), 1.0,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError('weights must sum to one; do not silently normalize')
    clean_scores = {k: finite(v, 'score') for k, v in scores.items()}
    if any(v < a or v > b for v in clean_scores.values()):
        raise ValueError('score outside registered range')
    return {
        'protocol': 'finite_support_census_v1',
        'kind': 'arithmetic_reference_not_a_semantic_certificate',
        'support_count': len(weights),
        'mean': math.fsum(clean_weights[k] * clean_scores[k] for k in weights),
        'confidence_method': 'not_applicable_census',
        'semantic_eta': None,
        'semantic_status': 'not_certified',
    }


def planned_calls(backgrounds: int, cells: int, models: int,
                  anchor_blocks_per_model: int = 0) -> dict:
    for name, value in [('backgrounds', backgrounds), ('cells', cells),
                        ('models', models)]:
        positive_int(value, name)
    if (isinstance(anchor_blocks_per_model, bool)
            or not isinstance(anchor_blocks_per_model, int)
            or anchor_blocks_per_model < 0):
        raise ValueError('anchor_blocks_per_model must be a nonnegative integer')
    core = backgrounds * cells * models
    anchors = anchor_blocks_per_model * cells * models
    return {'core_responses': core, 'anchor_responses': anchors,
            'total_before_retries_and_discussion': core + anchors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--j', required=True, type=int,
                        help='Hypothetical or actual allocated index; never reset a registry')
    parser.add_argument('--delta', type=float, default=.05)
    parser.add_argument('--tau', type=float, default=.20)
    args = parser.parse_args()
    try:
        rows = []
        for name, coeff in [('paired', PAIR), ('factorial', INTERACTION),
                            ('between_factorial', BETWEEN_INTERACTION)]:
            a, b = contrast_range(coeff)
            scale = (b - a) / 2
            for n in (32, 64, 128, 256, 512, 1024):
                r = scale * base_radius(args.j, n, args.delta)
                rows.append({'contrast': name, 'blocks': n, 'scale': scale,
                             'radius': r, 'strict_required_mean_eta_zero': args.tau + r,
                             'maximum_possible_lcb_eta_zero': b - r})
        print(json.dumps({'kind': 'planning_arithmetic_not_results',
                          'settings': vars(args), 'grid': rows}, indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
