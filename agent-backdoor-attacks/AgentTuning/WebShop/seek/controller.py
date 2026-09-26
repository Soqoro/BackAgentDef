"""Public-only diagnosis. No evaluator imports, model paths, environment or execution API."""
from .hypotheses import Hypotheses, mask_bank
from .preservation import apply_edits, check_window
from .replay import query
from .schemas import BehaviorPredicate, Invalid, PublicIncident, digest, outcome


def discover(snap, replay, victim, discussion, journal, config, method):
    incident = PublicIncident.from_dict(snap["public"])
    p = incident.to_dict()
    if replay.get("status") != "replay_valid" or replay.get("snapshot_hash") != snap["hash"]:
        return {"status": "replay_invalid", "candidate": None}
    if p["split"] not in ("development", "discovery") or p["track"] == "training_trace_smoke":
        raise Invalid("discovery cannot access holdout/training evidence")
    proposal = discussion.ask("State", incident, "proposal", semantic_preservation=method != "no_goal_preservation")
    proposed_count = len(proposal["spans"])
    spans = proposal["spans"][:config["budgets"]["max_spans"]]
    challenge = discussion.ask("Goal", incident, "challenge", spans=spans, proposal=proposal, semantic_preservation=method != "no_goal_preservation")
    revision = discussion.ask("State", incident, "revision", spans=spans, challenge=challenge,
                              semantic_preservation=method != "no_goal_preservation")
    # Revisions may delete candidates. New edits require a new challenge, never a bypass.
    if any(s not in spans for s in revision["spans"]):
        challenge = discussion.ask("Goal", incident, "rechallenge", spans=revision["spans"])
        spans = revision["spans"]
    rejected = {o["span_index"] for o in challenge["objections"] if o["change"] == "reject"}
    approved_indices = {o["span_index"] for o in challenge["objections"] if o["change"] == "keep"}
    rejected_texts = {digest(s) for i, s in enumerate(spans) if i not in approved_indices or i in rejected}
    valid = []
    for span in revision["spans"][:config["budgets"]["max_spans"]]:
        try:
            if method != "no_goal_preservation" and digest(span) in rejected_texts:
                raise Invalid("Goal objection requires exclusion")
            edited = apply_edits(incident, [span], semantic=method != "no_goal_preservation")
            check_window(victim, snap, edited)
            valid.append(span)
        except Invalid as exc:
            journal.emit("preservation_failure", {"span": span, "reason": str(exc), "phase": "discover"})
    journal.emit("challenge_resolution", {"objections": challenge["objections"], "before": spans, "after": valid,
                                          "changed": spans != valid, "why": revision["revision"]})
    coverage = {"proposed": proposed_count, "capped": proposed_count > config["budgets"]["max_spans"], "admissible": len(valid)}
    if not valid:
        return {"status": "no_valid_intervention", "candidate": None, "coverage": coverage}
    action_reply = discussion.ask("Action", incident, "predicate", spans=valid, challenge=challenge)
    if action_reply["predicate"] is None:
        return {"status": "inconclusive", "reason": "no_supported_behavior_predicate", "candidate": None, "coverage": coverage}
    try:
        predicate = BehaviorPredicate.from_dict(action_reply["predicate"], incident)
    except Invalid as exc:
        return {"status": "inconclusive", "reason": str(exc), "candidate": None, "coverage": coverage}
    journal.emit("predicate_frozen", predicate.to_dict())
    if method == "discussion_only":
        return {"status": "candidate", "candidate": [valid[0]], "predicate": predicate.to_dict(), "coverage": coverage,
                "explanation": proposal["prediction"], "alternatives": proposal["alternative"], "method_victim_queries": 0}
    model = Hypotheses(len(valid), config["budgets"]["interaction_order"])
    bank = mask_bank(len(valid))
    coverage.update(mask_bank_size=len(bank), possible_masks=2 ** len(valid), mask_bank_capped=len(bank) < 2 ** len(valid), hypotheses=len(model.subsets))
    probes, used = {}, set()

    def run_mask(mask, control=None):
        edits = [s for s, keep in zip(valid, mask) if not keep]
        edited = apply_edits(incident, edits, semantic=method != "no_goal_preservation")
        check_window(victim, snap, edited)
        result = query(victim, snap, edited, journal, "discover", config["budgets"]["discovery_victim"],
                       behavior_hash=digest(predicate.to_dict()), tag="discovery_input")
        score = outcome(predicate, result["action"], incident)
        probe_id = digest([snap["hash"], mask, predicate.to_dict()])
        record = {"mask": list(mask), "action": result["action"], "outcome": score}
        journal.emit("probe", {"probe_id": probe_id, "phase": "discover", **record})
        probes[probe_id] = record
        used.add(tuple(mask))
        if score["y"] is None:
            raise Invalid("unscorable discovery pair")
        if control is not None:
            model.control(control, score["y"])
        model.update(mask, score["y"], probe_id)
        return score["y"]

    try:
        original = run_mask((1,) * len(valid), 1)
        neutral = run_mask((0,) * len(valid), 0)
        if original <= neutral:
            return {"status": "no_localized_effect", "candidate": None, "coverage": coverage,
                    "predicate": predicate.to_dict(), "controls": {"original": original, "neutral": neutral}}
        for round_index in range(config["budgets"]["rounds"]):
            remaining = [list(z) for z in bank if tuple(z) not in used]
            if not remaining:
                break
            state = discussion.ask("State", incident, "next_experiment", spans=valid, probes=probes,
                                   mask_bank=remaining, round=round_index)
            goal = discussion.ask("Goal", incident, "probe_challenge", spans=valid, probes=probes,
                                  mask_bank=state["approved_masks"], proposal=state)
            approved = []
            for z in remaining if method == "fixed_probes" else state["approved_masks"]:
                if z not in remaining or (method not in ("no_goal_preservation", "fixed_probes") and z not in goal["approved_masks"]):
                    continue
                try:
                    edited = apply_edits(incident, [s for s, keep in zip(valid, z) if not keep],
                                         semantic=method != "no_goal_preservation")
                    check_window(victim, snap, edited)
                    approved.append(tuple(z))
                except Invalid:
                    pass
            selected = model.choose(approved, fixed=method == "fixed_probes")
            if selected is None:
                break
            run_mask(selected)
            discussion.ask("Action", incident, "evidence_update", spans=valid, probes=probes, predicate=predicate.to_dict())
        best = model.best()
        if not best:
            return {"status": "inconclusive", "candidate": None, "coverage": coverage, "reason": "null_hypothesis_preferred"}
        return {"status": "candidate", "candidate": [valid[i] for i in best], "predicate": predicate.to_dict(),
                "coverage": coverage, "posterior": [{"subset": list(s), "weight": w} for s, w in zip(model.subsets, model.weights())],
                "explanation": proposal["prediction"], "alternatives": proposal["alternative"],
                "method_victim_queries": len(probes), "control_rates": {"p0": model.rates()[0], "p1": model.rates()[1], "source": "discovery_controls_only"}}
    except Invalid as exc:
        return {"status": "inconclusive", "candidate": None, "coverage": coverage, "reason": str(exc)}
