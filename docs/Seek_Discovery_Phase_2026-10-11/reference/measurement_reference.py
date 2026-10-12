"""Independent arithmetic reference for the Seek handoff; not a repository runner.

No model execution, external dependencies, registry access, or research certificates.
The functions check numeric conventions only. Experiment validity, semantic meaning,
evidence freshness, and runtime reproducibility remain separate responsibilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def paper_radius(j: int, n: int, delta: float = 0.05, scale: float = 1.0) -> float:
    """Return the manuscript's raw-scale, time-uniform Hoeffding radius."""
    if isinstance(j, bool) or not isinstance(j, int) or j < 1:
        raise ValueError("j must be a positive integer")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be nonnegative and finite")
    log_term = (math.log(2.0) + math.log(j) + math.log(j + 1)
                + math.log(n) + math.log(n + 1) - math.log(delta))
    return scale * math.sqrt(2.0 * log_term / n)


def linear_range(coefficients: Sequence[float]) -> tuple[float, float]:
    """Range of a linear contrast whose cell outcomes each lie in [0,1]."""
    if not coefficients or any(not math.isfinite(c) for c in coefficients):
        raise ValueError("Provide nonempty finite coefficients")
    return sum(min(c, 0.0) for c in coefficients), sum(max(c, 0.0) for c in coefficients)


def radius_scale(coefficients: Sequence[float]) -> float:
    lower, upper = linear_range(coefficients)
    return (upper - lower) / 2.0


def linear_contrast(coefficients: Sequence[float], cells: Sequence[float]) -> float:
    linear_range(coefficients)
    if len(coefficients) != len(cells):
        raise ValueError("Each coefficient requires a cell outcome")
    if any(not math.isfinite(y) or not 0 <= y <= 1 for y in cells):
        raise ValueError("Cells must be finite outcomes in [0,1]")
    return sum(c * y for c, y in zip(coefficients, cells))


@dataclass(frozen=True)
class NumericBounds:
    implemented_lower: float
    implemented_upper: float
    semantic_lower: float | None
    semantic_upper: float | None


def numeric_bounds(mean: float, radius: float, eta: float | None = None) -> NumericBounds:
    """Numeric intervals only; eta=None must stay unknown, never become zero."""
    if not math.isfinite(mean) or not math.isfinite(radius) or radius < 0:
        raise ValueError("Mean and nonnegative radius must be finite")
    if eta is not None and (not math.isfinite(eta) or eta < 0):
        raise ValueError("eta must be nonnegative and finite, or None")
    lo, hi = mean - radius, mean + radius
    return NumericBounds(lo, hi, None if eta is None else lo - eta,
                         None if eta is None else hi + eta)


def strictly_exceeds(lower: float | None, threshold: float) -> bool:
    """Only the numeric comparison, not certification of an experiment."""
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    if lower is None:
        return False
    if not math.isfinite(lower):
        raise ValueError("Lower bound must be finite or None")
    return lower > threshold


def complete_support_mean(values: Sequence[float | None], weights: Sequence[float]) -> float:
    """Weighted mean of a complete prespecified support; no missing-cell deletion."""
    if not values or len(values) != len(weights):
        raise ValueError("Nonempty, aligned support and weights required")
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Missing/nonfinite values prevent a complete-support mean")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be nonnegative with positive finite mass")
    if not math.isfinite(sum(weights)):
        raise ValueError("Total weight must be finite")
    return sum(float(v) * w for v, w in zip(values, weights)) / sum(weights)


def wilson_interval(successes: int, studies: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Binomial Monte Carlo interval for independent study-level error indicators."""
    if (isinstance(studies, bool) or not isinstance(studies, int) or studies < 1
            or isinstance(successes, bool) or not isinstance(successes, int)
            or not 0 <= successes <= studies):
        raise ValueError("Require 0 <= successes <= studies, studies >= 1")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    p, z2 = successes / studies, z * z
    denom = 1.0 + z2 / studies
    center = (p + z2 / (2 * studies)) / denom
    half = z * math.sqrt(p * (1-p) / studies + z2 / (4 * studies * studies)) / denom
    return max(0.0, center-half), min(1.0, center+half)


def planned_budgets() -> dict[str, int]:
    return {
        "method_discovery_logical": 4 * 2 * 16,
        "method_evaluator_paired_logical": 4 * 3 * 12 * 2,
        "method_defender_logical_cap": 4 * 3 * 24,
        "observation_bridge_core": 4 * 2 * 2 * 2 * 3,
        "observation_competence_core": 4 * 2 * 3,
        "observation_anchors": 2 * 3,
        "query_panel_minimal_core": 4 * 3 * 3 * 3,
        "query_panel_expanded_core": 4 * 3 * 3 * 3 * 3,
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"purpose": "handoff arithmetic only; no experiment executed",
                      "claim10_radius_reference": paper_radius(10, 64),
                      "paired_scale": radius_scale([1,-1]),
                      "interaction_scale": radius_scale([1,-1,-1,1]),
                      "between_interaction_scale": radius_scale([1,-1,-1,1,-1,1,1,-1]),
                      "proposed_budget_caps": planned_budgets()}, indent=2))
