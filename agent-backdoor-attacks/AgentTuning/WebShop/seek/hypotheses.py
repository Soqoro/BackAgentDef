"""Exploratory likelihoods and information ordering; never confirmation evidence."""
from itertools import combinations
import math

from .schemas import Invalid


def mask_bank(n):
    masks = {(0,) * n, (1,) * n}
    for i in range(n):
        masks.add(tuple(int(j == i) for j in range(n)))
        masks.add(tuple(int(j != i) for j in range(n)))
    # Finite deterministic subset bank; exhaustive for small spaces only.
    for bits in range(min(2 ** n, 64)):
        masks.add(tuple((bits >> j) & 1 for j in range(n)))
    return sorted(masks)


class Hypotheses:
    def __init__(self, n, k=1):
        if not 1 <= n <= 16 or k not in (1, 2):
            raise Invalid("invalid sparse hypothesis space")
        self.subsets = [()] + [x for size in range(1, k + 1) for x in combinations(range(n), size)]
        self.logs = [0.0] * len(self.subsets)
        self.controls = {0: [], 1: []}
        self.observations = []

    def control(self, which, y):
        if y not in (0, 1):
            raise Invalid("unscorable control")
        self.controls[which].append(y)

    def rates(self):
        return tuple((1 + sum(self.controls[x])) / (2 + len(self.controls[x])) for x in (0, 1))

    def predictions(self, mask):
        p0, p1 = self.rates()
        return [(p0 + p1) / 2 if not subset else p0 + (p1 - p0) * math.prod(mask[j] for j in subset)
                for subset in self.subsets]

    def update(self, mask, y, probe_id):
        if y not in (0, 1) or any(p[2] == probe_id for p in self.observations):
            raise Invalid("unscorable or duplicate likelihood evidence")
        self.observations.append((tuple(mask), y, probe_id))
        # Recompute after control rates change; only actual scored probes enter.
        self.logs = [0.0] * len(self.subsets)
        for z, score, _ in self.observations:
            for i, mu in enumerate(self.predictions(z)):
                mu = min(1 - 1e-9, max(1e-9, mu))
                self.logs[i] += score * math.log(mu) + (1 - score) * math.log1p(-mu)

    def weights(self):
        m = max(self.logs)
        values = [math.exp(x - m) for x in self.logs]
        return [v / sum(values) for v in values]

    def information(self, mask, cost=1):
        weights, mus = self.weights(), self.predictions(mask)
        mean = sum(w * m for w, m in zip(weights, mus))
        return sum(w * (m - mean) ** 2 for w, m in zip(weights, mus)) / (cost + 1e-9)

    def choose(self, masks, fixed=False):
        if not masks:
            return None
        chosen = min(masks) if fixed else max(sorted(masks), key=self.information)
        return chosen if fixed or self.information(chosen) > 1e-12 else None

    def best(self):
        weights = self.weights()
        return self.subsets[max(range(len(weights)), key=weights.__getitem__)]
