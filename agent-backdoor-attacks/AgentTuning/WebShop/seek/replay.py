from .schemas import Invalid, digest
from .snapshot_io import validate_snapshot


def query(victim, snap, edited_input, journal, phase, budget, *, candidate_hash="none", behavior_hash="none", replicate=0, tag="probe"):
    r, p = snap["runtime"], snap["public"]
    key = {"snapshot": snap["hash"], "checkpoint": r["checkpoint_identity"],
           "template": r["template_hash"], "tokenizer": r["tokenizer"],
           "ids": victim.encode(victim.render(r["prefix_messages"], edited_input), r["generation"])[0],
           "generation": r["generation"], "backend": r["backend"], "dtype": r["dtype"],
           "role_prompt_version": "seek-roles-v1", "phase": phase, "cohort": p["split"],
           "candidate": candidate_hash, "behavior": behavior_hash, "replicate": replicate, "tag": tag}
    result = journal.call(key, "victim", phase, budget, lambda: victim.propose(snap, edited_input, r["generation"]))
    if result["simulated"] != p["simulated"]:
        raise Invalid("fake/real victim evidence segregation violation")
    return result


def no_edit_replay(victim, snap, journal, budget=1000):
    validate_snapshot(snap)
    p, r = snap["public"], snap["runtime"]
    if r["prefix_messages"] != p["history"]:
        raise Invalid("stale or modified pre-exposure history")
    prompt = victim.render(p["history"], p["policy_input"])
    ids, full = victim.encode(prompt, r["generation"])
    if prompt != r["serialized_prompt"] or ids != r["encoded_ids"] or full != r["full_ids"]:
        return {"case_id": p["case_id"], "status": "replay_invalid", "reason": "prompt_or_token_mismatch"}
    result = query(victim, snap, p["policy_input"], journal, "replay", budget, tag="no_edit")
    valid = result["encoded_ids"] == ids and result["action"] is not None and result["action"] == p["proposed_action"]
    return {"case_id": p["case_id"], "snapshot_hash": snap["hash"],
            "status": "replay_valid" if valid else "replay_invalid", "reason": "exact_ids_and_action" if valid else "action_mismatch",
            "raw_answer_equal": result["raw_response"] == snap["raw_response"],
            "raw_answer_hash": digest(result["raw_response"]), "context_truncated": full != ids}
