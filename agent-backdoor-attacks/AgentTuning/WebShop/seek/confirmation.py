"""Frozen, outcome-independent holdouts and conservative two-direction validation."""
from collections import defaultdict
import math
from pathlib import Path

from .preservation import apply_edits, check_window
from .replay import no_edit_replay, query
from .schemas import BehaviorPredicate, Invalid, PublicIncident, digest, outcome, predicate_applicable
from .storage import immutable_json, row_lock, read_json


def bound(pairs, alpha, family_M, threshold, expected_n):
    groups = defaultdict(list)
    invalid = 0
    for pair in pairs:
        values = pair["differences"]
        if not values or any(v is None or not -1 <= v <= 1 for v in values):
            invalid += 1
            continue
        # Repeated generations are averaged within each context, then by dependence group.
        groups[pair["dependence_group"]].append(sum(values) / len(values))
    n = len(groups)
    mean = sum(sum(v) / len(v) for v in groups.values()) / n if n else None
    radius = math.sqrt(2 * math.log(2 * family_M / alpha) / n) if n else None
    lcb = mean - radius if n else None
    return {"n": n, "expected_n": expected_n, "invalid_pairs": invalid, "mean_difference": mean,
            "radius": radius, "lcb": lcb, "threshold": threshold,
            "passes": not invalid and n >= expected_n and lcb is not None and lcb > threshold}


def candidate_edits(public, candidate, insertion=False):
    """Exact bytes in audited source regions; never rewrite a product/goal field."""
    text = public["policy_input"]
    spans, used = [], set()
    for part in candidate:
        needle = part["replacement"] if insertion else part["text"]
        replacement = part["text"] if insertion else part["replacement"]
        found = []
        start = 0
        while needle and (start := text.find(needle, start)) >= 0:
            end = start + len(needle)
            sources = [s for s in public["sources"] if s["start"] <= start and s["end"] >= end and s["kind"] == "narrative"]
            if sources and (start, end) not in used:
                found.append((start, end, sources[0]))
            start = end
        if not found or (not insertion and len(found) != 1):
            raise Invalid("candidate occurrence/neutral-slot eligibility mismatch")
        start, end, source = found[0]
        used.add((start, end))
        spans.append({"start": start, "end": end, "text": needle, "replacement": replacement, "source_fact": source["text"]})
    return spans


def select_holdouts(snapshots, candidate, split, count, channel, scope, checkpoint_identity, predicate=None):
    selected, groups = [], set()
    for snap in sorted(snapshots, key=lambda s: s["public"]["task_fingerprint"]):
        p = snap["public"]
        # Selection reads task/source eligibility, never proposal or attack outcome.
        if (p["split"] != split or p["channel"] != channel or p["scope"] != scope or
                snap["runtime"]["checkpoint_identity"] != checkpoint_identity or p["dependence_group"] in groups or
                p["track"] == "training_trace_smoke"):
            continue
        if predicate is not None and not predicate_applicable(predicate, p):
            continue
        insertion = split == "confirmation_insertion"
        if insertion and any(s["text"] in p["policy_input"] for s in candidate):
            continue
        try:
            edits = candidate_edits(p, candidate, insertion)
            apply_edits(PublicIncident.from_dict(p), edits)
        except Invalid:
            continue
        selected.append(p["case_id"])
        groups.add(p["dependence_group"])
        if len(selected) == count:
            break
    return selected


def freeze(discovery, discovery_snap, snapshots, protocol, path):
    if discovery.get("status") != "candidate":
        return None
    p, r = discovery_snap["public"], discovery_snap["runtime"]
    candidate = discovery["candidate"]
    public_parts = [{"text": s["text"], "replacement": s["replacement"]} for s in candidate]
    cohorts = {split: select_holdouts(snapshots, public_parts, split, protocol[n], p["channel"], p["scope"], r["checkpoint_identity"], discovery["predicate"])
               for split, n in (("confirmation_removal", "n_removal"), ("confirmation_insertion", "n_insertion"))}
    by_id = {s["public"]["case_id"]: s["public"] for s in snapshots}
    units = {split: [{"case_id": cid, "task": by_id[cid]["task_fingerprint"], "group": by_id[cid]["dependence_group"]} for cid in ids]
             for split, ids in cohorts.items()}
    record = {"schema_version": 1, "candidate": public_parts, "predicate": discovery["predicate"],
              "occurrence_rule": "unique_candidate_per_part; ordered_nonoverlapping_neutral_slots",
              "edit_protocol": "source_exact_narrative_replacement_v1", "channel": p["channel"], "scope": p["scope"],
              "checkpoint_identity": r["checkpoint_identity"], "checkpoint_alias": p["checkpoint_alias"],
              "discovery_task": p["task_fingerprint"], "discovery_group": p["dependence_group"],
              "cohorts": cohorts, "cohort_units": units, "protocol": protocol, "source": "natural_recorded_contexts" if not p["simulated"] else "simulated_constructed_contexts",
              "simulated": p["simulated"], "explanation": discovery.get("explanation"), "alternatives": discovery.get("alternatives")}
    record["hash"] = digest(record)
    immutable_json(path, record)
    return record


def register_family(run_root):
    root = Path(run_root)
    plan = read_json(root / "planned_rows.json")
    frozen_records = []
    for row in plan["rows"]:
        folder = root / row
        if not (folder / "discover.json").exists():
            raise Invalid("finish every registered discovery row before registering a confirmation family")
        if (folder / "confirm.json").exists():
            raise Invalid("cannot register a new family after confirmation")
        if (folder / "frozen_candidate.json").exists():
            frozen_records.append(read_json(folder / "frozen_candidate.json"))
    if not frozen_records:
        raise Invalid("no frozen candidates to register")
    sizes = {f["protocol"]["family_M"] for f in frozen_records}
    if len(sizes) != 1 or next(iter(sizes)) < len(frozen_records):
        raise Invalid("family M must cover all frozen candidates")
    registration = {"family_M": next(iter(sizes)), "frozen_hashes": sorted(f["hash"] for f in frozen_records)}
    immutable_json(root / "family_registration.json", registration)
    return registration


def claim_holdouts(ledger_root, frozen, family=None):
    """Persistent across rows/runs sharing the run root; revised candidates cannot reuse spent evidence."""
    root = Path(ledger_root) / "holdout_usage"
    candidate_key = digest({k: frozen[k] for k in ("candidate", "predicate", "protocol", "checkpoint_identity", "scope", "channel")})
    if family is not None:
        if frozen["hash"] not in family["frozen_hashes"] or family["family_M"] != frozen["protocol"]["family_M"]:
            raise Invalid("candidate not in the immutable preregistered family")
        candidate_key = digest(family)
    elif frozen["protocol"]["family_M"] > 1:
        raise Invalid("register-family before confirmation when family_M > 1")
    with row_lock(root):
        for units in frozen["cohort_units"].values():
            for unit in units:
                key = digest([frozen["checkpoint_identity"], unit["group"]])
                immutable_json(root / (key + ".json"), {"candidate_family": candidate_key})


def confirm(frozen, snapshots, victim, journal, config, method, overlap_status):
    if frozen["hash"] != digest({k: v for k, v in frozen.items() if k != "hash"}):
        raise Invalid("frozen candidate mutated")
    indexed = {s["public"]["case_id"]: s for s in snapshots}
    q = frozen["protocol"]
    contrasts = {}
    all_tasks, all_groups = set(), {frozen["discovery_group"]}
    for split, ids in frozen["cohorts"].items():
        insertion = split == "confirmation_insertion"
        pairs = []
        for case_id in ids:
            snap = indexed.get(case_id)
            if snap is None:
                pairs.append({"dependence_group": case_id, "differences": [None], "reason": "missing_context"})
                continue
            p = snap["public"]
            if p["split"] != split or p["task_fingerprint"] in all_tasks or p["dependence_group"] in all_groups or p["task_fingerprint"] == frozen["discovery_task"]:
                raise Invalid("confirmation task/dependence overlap or cohort mismatch")
            all_tasks.add(p["task_fingerprint"])
            all_groups.add(p["dependence_group"])
            incident = PublicIncident.from_dict(p)
            # Validation uses original evidence, never the edited state.
            pred = BehaviorPredicate(**frozen["predicate"])
            pair = {"case_id": case_id, "dependence_group": p["dependence_group"], "differences": [None]}
            try:
                replay = no_edit_replay(victim, snap, journal, config["budgets"]["replay_victim"])
                if replay["status"] != "replay_valid":
                    raise Invalid("replay_invalid")
                edits = candidate_edits(p, frozen["candidate"], insertion)
                edited = apply_edits(incident, edits, semantic=True)
                check_window(victim, snap, edited)
                arms = (edited, p["policy_input"]) if insertion else (p["policy_input"], edited)
                phase = "evaluator_confirm" if method == "discussion_only" or (method == "removal_only" and insertion) else "confirm"
                values = []
                for i, arm in enumerate(arms):
                    result = query(victim, snap, arm, journal, phase, config["budgets"]["confirmation_victim"],
                                   candidate_hash=frozen["hash"], behavior_hash=digest(frozen["predicate"]), tag=f"{split}:arm-{i}")
                    score = outcome(pred, result["action"], incident)
                    values.append(score["y"])
                    journal.emit("confirmation_arm", {"case_id": case_id, "split": split, "arm": i, "score": score, "phase": phase})
                pair["differences"] = [None if None in values else values[0] - values[1]]
                pair["reason"] = "unscorable" if None in values else "scored"
            except Invalid as exc:
                pair["reason"] = str(exc)
            except Exception as exc:
                pair["reason"] = "backend_failure:" + type(exc).__name__
            pairs.append(pair)
        label = "insertion" if insertion else "removal"
        contrasts[label] = {"pairs": pairs, **bound(pairs, q["alpha"], q["family_M"], q["tau_" + label], q["n_" + label])}
    enough = all(contrasts[x]["n"] >= q["n_" + x] for x in ("removal", "insertion"))
    valid = all(contrasts[x]["passes"] for x in ("removal", "insertion")) and overlap_status == "excluded"
    invalid = any(contrasts[x]["invalid_pairs"] for x in ("removal", "insertion"))
    return {"status": "validated" if valid else "inconclusive" if enough or invalid else "insufficient_holdout",
            "evaluator_functional_validated": valid, "method_claim": contrasts["removal"]["passes"] if method == "removal_only" else valid,
            "contrasts": contrasts, "family_M": q["family_M"], "alpha": q["alpha"], "frozen_hash": frozen["hash"],
            "overlap_status": overlap_status, "exact_recovery": None, "ground_truth": "unknown",
            "scope_limit": "independent task/product groups within one checkpoint; no cross-checkpoint guarantee",
            "simulated": frozen["simulated"]}
