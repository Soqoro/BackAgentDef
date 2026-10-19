"""Independent planning arithmetic; not production Seek or experiment evidence."""
from __future__ import annotations
import math
from typing import Mapping, Sequence


def _integer(x: int, name: str, minimum: int = 0) -> int:
    if type(x) is not int or x < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return x


def paired_difference(values: Sequence[int | None]) -> int | None:
    if len(values) != 2:
        raise ValueError('exactly two ordered arms are required')
    for x in values:
        if x is not None and (type(x) is not int or x not in (0, 1)):
            raise ValueError('an outcome must be binary or unscorable (None)')
    if any(x is None for x in values):
        return None
    return values[0] - values[1]


def prediction_loss(predictions: Sequence[float | None], values: Sequence[int | None]) -> float | None:
    if len(predictions) != 2:
        raise ValueError('exactly two predictions are required')
    paired_difference(values)
    for p in predictions:
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError('predictions must lie in [0,1] or be unknown')
    if any(p is None for p in predictions) or any(v is None for v in values):
        return None
    return sum((p-v)**2 for p, v in zip(predictions, values))/2


def radius(j: int, n: int, delta: float = 0.05, scale: float = 1.0) -> float:
    _integer(j, 'claim index', 1)
    _integer(n, 'sample count', 1)
    if not 0 < delta < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid error level or contrast scale')
    log_term = math.log(2) + math.log(j) + math.log(j+1) + math.log(n) + math.log(n+1) - math.log(delta)
    return scale * math.sqrt(2 * log_term/n)


def lower_bound(mean: float, j: int, n: int, *, eta: float | None, scale: float = 1.0) -> float | None:
    if not math.isfinite(mean) or not -scale <= mean <= scale:
        raise ValueError('mean is outside the declared contrast range')
    r = radius(j, n, scale=scale)
    if eta is None:
        return None  # semantic uncertainty is not silently zero
    if not math.isfinite(eta) or eta < 0:
        raise ValueError('invalid semantic discrepancy bound')
    return mean-r-eta


def alarm_rate(alarms: int, exposures: int, *, alarm_rule_defined: bool) -> float | None:
    _integer(alarms, 'alarms'); _integer(exposures, 'exposures')
    if alarms > exposures:
        raise ValueError('alarms exceed opportunities')
    if not alarm_rule_defined or exposures == 0:
        return None
    return alarms/exposures


def paired_call_budget(blocks: int, arms: int, policies: int) -> int:
    for x, name in [(blocks, 'blocks'), (arms, 'arms'), (policies, 'policies')]:
        _integer(x, name)
    return blocks * arms * policies


def method_budget(opportunities_per_condition: int, conditions: int, methods: int,
                  discovery_cap: int, evaluation_pairs: int, evaluation_components: int) -> Mapping[str, int]:
    xs=[opportunities_per_condition, conditions, methods, discovery_cap, evaluation_pairs, evaluation_components]
    for i, x in enumerate(xs): _integer(x, f'argument {i}')
    count=opportunities_per_condition*conditions*methods
    return {'potential_investigations':count,
            'discovery_logical_ceiling':count*discovery_cap,
            'evaluation_logical_ceiling':count*evaluation_pairs*2*evaluation_components}
