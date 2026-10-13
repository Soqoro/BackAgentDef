"""Small CPU-only arithmetic/logic reference for the proposed Seek upgrade.

Not a BackAgentDef implementation; no LLM, renderer, semantic verifier or causal
recovery is implemented here. Prediction weights are exploratory scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


def _finite(value: float, name: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def normalize(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("nonempty hypothesis family required")
    clean = {k: _finite(v, "weight") for k, v in weights.items()}
    if any(v < 0 for v in clean.values()) or sum(clean.values()) <= 0:
        raise ValueError("weights must be nonnegative with positive total")
    total = math.fsum(clean.values())
    return {k: v / total for k, v in clean.items()}


def arm_loss(prediction: Sequence[float], outcome: Sequence[float | None]) -> float | None:
    if len(prediction) != len(outcome) or not prediction:
        raise ValueError("nonempty aligned arms required")
    losses: list[float] = []
    for p, y in zip(prediction, outcome):
        p = _finite(p, "prediction")
        if not 0 <= p <= 1:
            raise ValueError("probabilities must be in [0,1]")
        if y is None:
            continue
        y = _finite(y, "outcome")
        if y not in (0, 1):
            raise ValueError("this reference accepts binary outcomes or None")
        losses.append((p - y) ** 2)
    return math.fsum(losses) / len(losses) if losses else None


@dataclass(frozen=True)
class Update:
    weights: dict[str, float]
    losses: dict[str, float | None]
    status: str


class EvidenceReducer:
    """Illustrative update only; evidence keys identify reusable deterministic facts.

    A production implementation must construct keys from exact inputs/runtime and
    separately implement sampling semantics. A new event ID is not such a key.
    """

    def __init__(self, weights: Mapping[str, float], learning_rate: float = 1.0):
        self.weights = normalize(weights)
        self.learning_rate = _finite(learning_rate, "learning rate")
        if self.learning_rate < 0:
            raise ValueError("nonnegative learning rate required")
        self.seen: set[str] = set()

    def update(self, evidence_key: str, predictions: Mapping[str, Sequence[float]],
               outcomes: Sequence[float | None]) -> Update:
        if not evidence_key:
            raise ValueError("evidence key required")
        if set(predictions) != set(self.weights):
            raise ValueError("missing or extra hypothesis predictions")
        losses = {h: arm_loss(predictions[h], outcomes) for h in self.weights}
        if evidence_key in self.seen:
            return Update(dict(self.weights), losses, "duplicate_no_update")
        self.seen.add(evidence_key)
        if all(loss is None for loss in losses.values()):
            return Update(dict(self.weights), losses, "unscorable_no_update")
        # All hypotheses see the same scorable arm subset. This is discovery
        # ranking only; complete-pair confirmation has a separate missingness rule.
        logs = {
            h: math.log(max(self.weights[h], 1e-300)) - self.learning_rate * float(losses[h])
            for h in self.weights
        }
        peak = max(logs.values())
        self.weights = normalize({h: math.exp(v - peak) for h, v in logs.items()})
        return Update(dict(self.weights), losses, "exploratory_updated")


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    treatment_key: str
    predictions: Mapping[str, Sequence[float]]
    logical_cost: int
    valid: bool = True


def disagreement(experiment: Experiment, weights: Mapping[str, float]) -> float:
    if experiment.logical_cost <= 0:
        raise ValueError("logical response cost must be positive")
    w = normalize(weights)
    if set(experiment.predictions) != set(w):
        raise ValueError("unknown predictions require coverage review, not zero variance")
    lengths = {len(p) for p in experiment.predictions.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("prediction vectors must align")
    total = 0.0
    for arm in range(next(iter(lengths))):
        values = {h: _finite(experiment.predictions[h][arm], "prediction") for h in w}
        if any(not 0 <= p <= 1 for p in values.values()):
            raise ValueError("probabilities must be in [0,1]")
        mu = math.fsum(w[h] * values[h] for h in w)
        total += math.fsum(w[h] * (values[h] - mu) ** 2 for h in w)
    return total / experiment.logical_cost


@dataclass(frozen=True)
class Selection:
    experiment_id: str | None
    status: str
    utility: float | None


def select(experiments: Sequence[Experiment], weights: Mapping[str, float],
           seen_treatments: set[str], remaining_calls: int,
           coverage_turn: bool = False, coverage_order: Sequence[str] = ()) -> Selection:
    if remaining_calls < 0:
        raise ValueError("nonnegative budget required")
    eligible = [e for e in experiments if e.valid and e.logical_cost <= remaining_calls
                and e.treatment_key not in seen_treatments]
    scored = [(e, disagreement(e, weights)) for e in eligible]
    # Coverage uses structural separability, not near-zero posterior/score mass.
    # A downweighted rule must not become unreachable merely because its score is small.
    separating = [(e, s) for e, s in scored
                  if len({tuple(v) for v in e.predictions.values()}) > 1]
    if not separating:
        return Selection(None, "no_valid_unseen_separating_test", None)
    if coverage_turn or max(s for _, s in separating) <= 1e-15:
        order = {key: i for i, key in enumerate(coverage_order)}
        e, score = min(separating, key=lambda es: (order.get(es[0].experiment_id, len(order)),
                                                  es[0].experiment_id))
        return Selection(e.experiment_id, "coverage_selected" if coverage_turn else "coverage_fallback", score)
    e, score = min(separating, key=lambda es: (-es[1], es[0].experiment_id))
    return Selection(e.experiment_id, "disagreement_selected", score)


def equivalence_classes(experiments: Sequence[Experiment], hypotheses: Sequence[str]) -> list[list[str]]:
    """Exact prediction equivalence over the supplied bank, not semantic equivalence."""
    classes: dict[tuple[float, ...], list[str]] = {}
    for h in hypotheses:
        sig = tuple(float(p) for e in experiments for p in e.predictions[h])
        classes.setdefault(sig, []).append(h)
    return sorted((sorted(v) for v in classes.values()), key=lambda group: group[0])


def pair_summary(pairs: Sequence[tuple[int | None, int | None]]) -> dict[str, object]:
    for pair in pairs:
        if len(pair) != 2 or any(y not in (0, 1, None) for y in pair):
            raise ValueError("binary pair outcomes or None required")
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    counts = {"high_high": 0, "high_low": 0, "low_high": 0, "low_low": 0}
    for a, b in valid:
        counts[("high" if a else "low") + "_" + ("high" if b else "low")] += 1
    return {
        "pairs_total": len(pairs), "complete_pairs": len(valid),
        "missing_pairs": len(pairs) - len(valid), "joint_counts": counts,
        "complete_pair_mean": sum(a - b for a, b in valid) / len(valid) if valid else None,
        "scope": "descriptive_complete_pairs_only_not_confirmation",
    }


def paper_radius(j: int, n: int, delta: float = .05, range_scale: float = 1.) -> float:
    if isinstance(j, bool) or isinstance(n, bool) or not isinstance(j, int) or not isinstance(n, int):
        raise ValueError("integer j and n required")
    if j < 1 or n < 1 or not 0 < delta < 1:
        raise ValueError("j,n >=1 and 0<delta<1 required")
    scale = _finite(range_scale, "range scale")
    if scale <= 0:
        raise ValueError("positive range scale required")
    return scale * math.sqrt(2. / n * math.log(2 * j * (j + 1) * n * (n + 1) / delta))


def certificate(mean: float, radius: float, threshold: float, eta: float | None) -> dict[str, object]:
    mean = _finite(mean, "mean")
    radius = _finite(radius, "radius")
    threshold = _finite(threshold, "threshold")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    impl = mean - radius
    if eta is not None:
        eta = _finite(eta, "eta")
        if eta < 0:
            raise ValueError("eta must be nonnegative")
    semantic = None if eta is None else impl - eta
    return {
        "implemented_lower": impl, "implemented_pass": impl > threshold,
        "semantic_lower": semantic,
        "semantic_pass": None if semantic is None else semantic > threshold,
        "assumptions_verified_by_this_function": False,
    }


def noiseless_identify(predictions: Mapping[str, Mapping[str, int]], true_rule: str) -> dict[str, object]:
    """Idealized greedy version-space demo, NOT the production soft search.

    All predictions are supplied known-rule fixture data. Pairwise separability
    is checked operationally. No theorem about an LLM-generated graph is implied.
    """
    if true_rule not in predictions or not predictions:
        raise ValueError("realizable known-rule fixture required")
    test_sets = [set(v) for v in predictions.values()]
    if any(s != test_sets[0] for s in test_sets):
        raise ValueError("aligned test bank required")
    if any(y not in (0, 1) for v in predictions.values() for y in v.values()):
        raise ValueError("binary rule predictions required")
    live = set(predictions)
    trace = []
    while len(live) > 1:
        alternatives = []
        for test in sorted(test_sets[0]):
            zero = {h for h in live if predictions[h][test] == 0}
            one = live - zero
            if zero and one:
                alternatives.append((max(len(zero), len(one)), test, zero, one))
        if not alternatives:
            return {"status": "equivalence_class", "rules": sorted(live), "trace": trace}
        _, test, zero, one = min(alternatives, key=lambda x: (x[0], x[1]))
        label = predictions[true_rule][test]
        before = len(live)
        live = one if label else zero
        trace.append({"test": test, "label": label, "before": before, "after": len(live)})
    return {"status": "identified_within_fixture", "rules": sorted(live), "trace": trace}


def demo() -> dict[str, object]:
    names = ["literal", "narrow_concept", "broad_category", "general_tendency", "low_tendency"]
    lexical = {
        "literal": (.9, .1), "narrow_concept": (.9, .9),
        "broad_category": (.9, .9), "general_tendency": (.9, .9), "low_tendency": (.1, .1),
    }
    boundary = {
        "literal": (.9, .1), "narrow_concept": (.9, .1),
        "broad_category": (.9, .9), "general_tendency": (.9, .9), "low_tendency": (.1, .1),
    }
    reducer = EvidenceReducer(dict.fromkeys(names, 1.))
    updates = []
    for i in range(4):
        result = reducer.update(f"distinct_fixture_pair_{i}", lexical, (1, 1))
        updates.append({"weights": result.weights, "losses": result.losses, "status": result.status})
    bank = [Experiment("repeat_wording", "already_tested", lexical, 2),
            Experiment("generic_sibling_boundary", "unseen_boundary", boundary, 2)]
    choice = select(bank, reducer.weights, {"already_tested"}, remaining_calls=4)
    return {
        "simulated": True, "not_real_model_evidence": True,
        "input": "Four fixture pairs with outcome (1,1); no semantic verification.",
        "joint_summary": pair_summary([(1, 1)] * 4), "updates": updates,
        "selection": {"experiment": choice.experiment_id, "status": choice.status, "utility": choice.utility},
        "interpretation": "A toy rule's predictions were weakened; no trigger or semantic certificate is established.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2, allow_nan=False))
