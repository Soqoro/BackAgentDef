"""Bounded evidence display; never changes the search family, weights or choices."""
from ..schemas import digest


def display(family, events, weights, limit=8):
    rules = family['hypotheses']
    by_id = {r['id']: r for r in rules}
    if not events:
        ids = sorted(by_id, key=lambda k: (-weights[k], k))[:limit]
        return [by_id[k] for k in ids], None
    losses = events[-1]['losses']
    scored = [k for k in by_id if losses.get(k) is not None]
    unknown = sorted(k for k in by_id if losses.get(k) is None)
    best = min((losses[k] for k in scored), default=None)
    winners = sorted((k for k in scored if losses[k] == best), key=lambda k: (by_id[k]['ast']['op'], k))
    chosen = []
    def add(k):
        if k not in chosen and len(chosen) < limit: chosen.append(k)
    # Show different rule forms among tied best predictions, not only high weights.
    seen_ops = set()
    for k in winners:
        op = by_id[k]['ast']['op']
        if op not in seen_ops and len(chosen) < 3:
            add(k); seen_ops.add(op)
    for k in sorted(scored, key=lambda k: (-losses[k], k))[:2]: add(k)
    for k in unknown[:1]: add(k)
    for k in winners: add(k)
    for k in sorted(by_id, key=lambda k: (-weights[k], k)): add(k)
    groups = {}
    for k in scored:
        signature = digest([(e['predictions'][k], e['losses'][k]) for e in events])
        groups.setdefault(signature, []).append(k)
    summary = dict(
        selection='algorithmic evidence display: tied best rule forms, high loss, unassessed, then weights',
        total_rules=len(rules), omitted_rules=len(rules)-len(chosen),
        scored_rules=len(scored), unassessed_rules=len(unknown),
        latest_loss_range=None if not scored else [min(losses[k] for k in scored), max(losses[k] for k in scored)],
        equal_observed_prediction_groups=[dict(visible_ids=sorted(set(g)&set(chosen)), omitted_count=len(set(g)-set(chosen)))
            for g in groups.values() if len(g)>1 and set(g)&set(chosen)],
        interpretation='Equal predictions/losses on performed probes do not distinguish mechanisms. Null losses are unassessed. Display omissions are not refutations.',
        search_unchanged=True)
    return [by_id[k] for k in chosen], summary
