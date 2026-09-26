from .confirmation import candidate_edits
from .preservation import apply_edits, check_window
from .replay import no_edit_replay, query
from .schemas import BehaviorPredicate, Invalid, PublicIncident, digest, outcome


def make_signature(frozen, confirmation, diagnostic_calls):
    if not confirmation["evaluator_functional_validated"]:
        return None
    return {"schema_version": 1, "frozen": frozen, "confirmation_hash": digest(confirmation),
            "diagnostic_calls": diagnostic_calls, "simulated": frozen["simulated"]}


def reuse(signature, snapshots, victim, journal, budget):
    frozen = signature["frozen"]
    if signature["schema_version"] != 1 or victim.identity != frozen["checkpoint_identity"]:
        return {"status": "revoked", "records": [], "reason": "incompatible_schema_or_checkpoint", "simulated": signature["simulated"]}
    revoked, records = False, []
    for snap in snapshots:
        p = snap["public"]
        if p["split"] != "reuse":
            continue
        match = (p["channel"] == frozen["channel"] and p["scope"] == frozen["scope"] and
                 snap["runtime"]["checkpoint_identity"] == frozen["checkpoint_identity"] and
                 all(x["text"] in p["policy_input"] for x in frozen["candidate"]))
        row = {"case_id": p["case_id"], "matched": match, "false_match": None, "saved_diagnostic_calls": 0}
        if match and not revoked:
            try:
                audit = no_edit_replay(victim, snap, journal)
                if audit["status"] != "replay_valid":
                    raise Invalid("replay_invalid")
                incident = PublicIncident.from_dict(p)
                edits = candidate_edits(p, frozen["candidate"])
                edited = apply_edits(incident, edits)
                check_window(victim, snap, edited)
                pred = BehaviorPredicate(**frozen["predicate"])
                values = []
                for arm in (p["policy_input"], edited):
                    result = query(victim, snap, arm, journal, "reuse", budget, candidate_hash=frozen["hash"], behavior_hash=digest(frozen["predicate"]))
                    values.append(outcome(pred, result["action"], incident)["y"])
                agrees = None not in values and values[0] > values[1]
                row["false_match"] = not agrees if None not in values else None
                row["saved_diagnostic_calls"] = max(0, signature["diagnostic_calls"] - 3) if agrees else 0
                revoked = not agrees
            except Invalid:
                revoked = True
                row["false_match"] = None
        row["revoked"] = revoked
        records.append(row)
    return {"status": "revoked" if revoked else "complete", "records": records,
            "execution_effect": "none; Shield unchanged",
            "savings_basis": "estimate using prior observed discovery calls minus two contrast arms and one replay; not a counterfactual measurement", "simulated": signature["simulated"]}
