#!/usr/bin/env python3
"""Independent standard-library check of the manuscript's Seek confidence rule.

This is NOT the repository's Seek implementation, an experiment runner, or a
semantic-validity verifier. Its output is an algebraic budget calculation.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def radius(j: int, n: int, delta: float = 0.05) -> float:
    """Two-sided, all-claim/all-time Hoeffding-union radius in the paper."""
    _positive_int(j, "j")
    _positive_int(n, "n")
    delta = _finite(delta, "delta")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    log_term = (math.log(2.0) + math.log(j) + math.log(j + 1)
                + math.log(n) + math.log(n + 1) - math.log(delta))
    return math.sqrt(2.0 * log_term / n)


@dataclass(frozen=True)
class Bounds:
    n: int
    mean_difference: float
    radius: float
    implemented_lcb: float
    implemented_ucb: float
    implemented_exceeds_threshold: bool
    semantic_eta: float | None
    semantic_lcb: float | None
    semantic_ucb: float | None
    semantic_exceeds_threshold_conditional: bool | None


def evaluate(scores: Iterable[float], j: int = 1, delta: float = 0.05,
             tau: float = 0.20, eta: float | None = None) -> Bounds:
    """Compute bounds; eta's justification is the caller's responsibility.

    A numeric eta only computes a CONDITIONAL semantic result. This utility
    cannot certify independence, renderer fidelity, or evidence freshness.
    Empty data raise ValueError rather than creating a zero-effect observation.
    """
    tau = _finite(tau, "tau")
    if not -1.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [-1,1]")
    if eta is not None:
        eta = _finite(eta, "eta")
        if eta < 0:
            raise ValueError("eta must be nonnegative")
    values = [_finite(v, "score") for v in scores]
    if not values or any(v < -1.0 or v > 1.0 for v in values):
        raise ValueError("scores must be a nonempty collection in [-1,1]")
    n = len(values)
    mean = math.fsum(values) / n
    r = radius(j, n, delta)
    lower, upper = mean - r, mean + r
    sl = None if eta is None else lower - eta
    su = None if eta is None else upper + eta
    return Bounds(n, mean, r, lower, upper, lower > tau, eta, sl, su,
                  None if sl is None else sl > tau)


def minimum_pairs(effect: float, j: int = 1, delta: float = 0.05,
                  tau: float = 0.20, eta: float = 0.0,
                  criterion: str = "observed_crossing",
                  max_pairs: int = 1_000_000) -> int | None:
    """Find a mathematical boundary, NOT a predicted stopping time.

    observed_crossing: hypothetical observed mean effect - r - eta > tau.
    theorem_sufficient: hypothetical true semantic effect has g>0 and 2r<g.
    """
    _positive_int(max_pairs, "max_pairs")
    effect, tau, eta = (_finite(effect, "effect"), _finite(tau, "tau"),
                        _finite(eta, "eta"))
    if not -1 <= effect <= 1 or not -1 <= tau <= 1 or eta < 0:
        raise ValueError("effects/threshold must be in [-1,1]; eta >= 0")
    radius(j, 1, delta)  # Validate independently of the early return below.
    if criterion == "observed_crossing":
        gap, multiplier = effect - tau - eta, 1.0
    elif criterion == "theorem_sufficient":
        gap, multiplier = effect - tau - 2.0 * eta, 2.0
    else:
        raise ValueError("unknown criterion")
    if gap <= 0 or multiplier * radius(j, max_pairs, delta) >= gap:
        return None
    lo, hi = 1, max_pairs
    while lo < hi:
        mid = (lo + hi) // 2
        if multiplier * radius(j, mid, delta) < gap:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="conditional calculation only; does not justify eta")
    args = parser.parse_args()
    try:
        rows = []
        for n in (8, 16, 32, 64, 128, 256, 512, 1024):
            r = radius(args.j, n, args.delta)
            rows.append({"pairs": n, "two_arm_victim_calls": 2 * n,
                         "radius": r,
                         "minimum_observed_effect_strictly_above": args.tau + r + args.eta,
                         "max_possible_lcb": 1.0 - r - args.eta})
        crossing = []
        for effect in (0.4, 0.6, 0.8):
            kw = dict(effect=effect, j=args.j, delta=args.delta, tau=args.tau, eta=args.eta)
            crossing.append({"hypothetical_effect": effect,
                             "observed_crossing_pairs": minimum_pairs(**kw),
                             "theorem_sufficient_pairs": minimum_pairs(
                                 **kw, criterion="theorem_sufficient")})
        print(json.dumps({"kind": "algebraic_budget_not_experimental_results",
                          "settings": vars(args), "grid": rows,
                          "hypothetical_boundaries": crossing}, indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
