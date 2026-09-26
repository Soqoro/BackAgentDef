"""Counts retain missingness and distinguish evidence units and call categories."""
from collections import Counter, defaultdict


def call_metrics(records):
    by_phase = defaultdict(lambda: {"attempted": 0, "successful": 0, "cached": 0, "logical": 0, "unique_inputs": 0,
                                    "interrupted_or_failed": 0, "latency_seconds": 0.0})
    inputs, finished, attempts = defaultdict(set), set(), {}
    roles = defaultdict(lambda: {"attempts": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0, "usage_missing": 0, "cached": 0})
    for event in records:
        d, kind = event["data"], event["kind"]
        if kind in ("call_attempt", "call_complete", "cache_hit"):
            key = d["category"] + ":" + d["phase"]
            m = by_phase[key]
            if kind == "call_attempt":
                m["attempted"] += 1
                inputs[key].add(d["input_hash"])
                attempts[event["id"]] = key
            elif kind == "call_complete":
                m["successful"] += 1
                m["latency_seconds"] += d["latency_seconds"]
                finished.add(d["attempt"])
            else:
                m["cached"] += 1
        if kind == "call_attempt" and d["category"] == "defender":
            m = roles[d.get("role") or "unknown"]
            m["attempts"] += 1
            m["retries"] += int(d.get("retry", 0) > 0)
        if kind == "cache_hit" and d["category"] == "defender":
            roles[d.get("role") or "unknown"]["cached"] += 1
        if kind == "call_complete" and d["category"] == "defender":
            usage = d["result"].get("usage", {})
            m = roles[d.get("role") or "unknown"]
            if usage.get("usage_reported"):
                m["input_tokens"] += usage.get("input_tokens", 0)
                m["output_tokens"] += usage.get("output_tokens", 0)
            else:
                m["usage_missing"] += 1
    for attempt, key in attempts.items():
        if attempt not in finished:
            by_phase[key]["interrupted_or_failed"] += 1
    for key, m in by_phase.items():
        m["logical"] = m["attempted"] + m["cached"]
        m["unique_inputs"] = len(inputs[key])
        m["actual_completed"] = m["successful"]
        m["actual_execution_unknown"] = m["interrupted_or_failed"]
    return {"calls": dict(by_phase), "defender_roles": dict(roles), "monetary_estimate_usd": None,
            "pricing_provenance": None, "AER": None, "AER_definition": "average episode reward; not probe deviation rate"}


def denominator_summary(snapshots, replay, discovery, confirmation):
    p = [s["public"] for s in snapshots]
    tracks = Counter(x["track"] for x in p)
    return {"captured_contexts": len(p), "captured_tasks": len({x["task_fingerprint"] for x in p}),
            "raw_audits": tracks["raw_audit"], "shield_incidents": tracks["shield_incident"],
            "training_trace_smoke": tracks["training_trace_smoke"],
            "replays": len(replay), "valid_replays": sum(x["status"] == "replay_valid" for x in replay),
            "investigations": int(discovery is not None),
            "editable_candidates": discovery.get("coverage", {}).get("admissible") if discovery else None,
            "completed_confirmations": int(confirmation is not None),
            "validated_signatures": int(confirmation["evaluator_functional_validated"]) if confirmation else None,
            "exact_recovery": None, "false_attribution": None, "false_validation": None,
            "reason_for_NA": "unknown evaluator truth or unmeasured denominator; never inferred as zero"}
