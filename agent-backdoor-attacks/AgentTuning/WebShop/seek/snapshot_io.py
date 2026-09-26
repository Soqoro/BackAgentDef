"""Immutable pre-call capture and explicit, non-executable legacy import."""
import json
from pathlib import Path

from .schemas import Invalid, PublicIncident, canonical, digest
from .storage import immutable_json

RUNTIME_FIELDS = {"system", "template_id", "template_hash", "template_source_hash", "tokenizer",
                  "generation", "backend", "dtype", "checkpoint_identity", "encoded_ids", "full_ids",
                  "serialized_prompt", "prefix_messages", "reset_boundary", "environment_fingerprint",
                  "filter_fingerprint", "catalogue_fingerprint", "exposure_knowledge", "structured_state", "frozen_contract"}


def snapshot(public, runtime, response, executed_action=None, shield_report=None):
    data = {"public": public.to_dict(), "runtime": json.loads(canonical(runtime)),
            "raw_response": response, "executed_action": executed_action,
            "shield_report": shield_report or {}, "capture_stage": "before_output_intervention"}
    data["hash"] = digest(data)
    return data


def validate_snapshot(data, complete=True):
    if set(data) != {"public", "runtime", "raw_response", "executed_action", "shield_report", "capture_stage", "hash"}:
        raise Invalid("incomplete/unknown snapshot fields")
    if data["hash"] != digest({k: v for k, v in data.items() if k != "hash"}):
        raise Invalid("snapshot hash mismatch")
    public = PublicIncident.from_dict(data["public"])
    r = data["runtime"]
    if set(r) != RUNTIME_FIELDS or any(r[k] is None for k in RUNTIME_FIELDS):
        raise Invalid("incomplete runtime capture")
    if data["capture_stage"] != "before_output_intervention":
        raise Invalid("post-intervention record cannot certify replay")
    if complete and not isinstance(data["raw_response"], str):
        raise Invalid("missing unmodified response")
    if r["reset_boundary"] not in ("episode_reset", "candidate_relative_prefix"):
        raise Invalid("stale KV/history or unknown reset boundary")
    if not r["encoded_ids"] or any(type(i) is not int for i in r["encoded_ids"] + r["full_ids"]):
        raise Invalid("missing/invalid consumed token IDs")
    if r["generation"].get("do_sample") is not False:
        raise Invalid("first-version replay requires greedy decoding")
    return public


def save_snapshot(root, data):
    validate_snapshot(data)
    immutable_json(Path(root) / "snapshots" / (data["public"]["case_id"] + ".json"), data)


def import_legacy(path):
    """Read trailing-comma JSON records without eval; never fabricate runtime metadata.

    Output deliberately contains no legacy filenames/IDs or training conversations.
    It is integration inventory, not detector or held-out evidence.
    """
    text = Path(path).read_text()
    decoder, offset, records = json.JSONDecoder(), 0, []
    if text.lstrip().startswith("["):
        values = json.loads(text)
    else:
        values = []
        while offset < len(text):
            while offset < len(text) and text[offset] in " \r\n\t,":
                offset += 1
            if offset == len(text):
                break
            value, offset = decoder.raw_decode(text, offset)
            values.append(value)
    for value in values:
        if not isinstance(value, dict):
            raise Invalid("legacy record is not an object")
        records.append({"record_hash": digest(value), "track": "training_trace_smoke",
                        "status": "incomplete", "replay_eligible": False,
                        "reason": "missing pre-call tokens/template/raw pre-intervention proposal; exposure unknown"})
    return records
