"""Source equality is authoritative; role agreement cannot waive these checks."""
import re
from .schemas import Invalid, SPAN, validate


def check_sources(public):
    text = public["policy_input"]
    end = 0
    for source in sorted(public["sources"], key=lambda s: s["start"]):
        if source["start"] != end or source["end"] <= end or text[source["start"]:source["end"]] != source["text"]:
            raise Invalid("source layout has gaps, overlap or incorrect offsets")
        end = source["end"]
    if end != len(text):
        raise Invalid("source layout does not cover full policy input")


def apply_edits(incident, spans, *, semantic=True):
    p = incident.to_dict()
    check_sources(p)
    text, last = p["policy_input"], 0
    result = []
    for span in sorted(spans, key=lambda s: s["start"]):
        validate(span, SPAN)
        start, end = span["start"], span["end"]
        if start < last or end <= start or text[start:end] != span["text"] or not span["text"]:
            raise Invalid("source_offset_mismatch")
        covering = [s for s in p["sources"] if s["start"] <= start and s["end"] >= end]
        if len(covering) != 1 or covering[0]["kind"] != "narrative":
            raise Invalid("hard_identity_action_or_attribute_edit")
        if span["source_fact"] != covering[0]["text"]:
            raise Invalid("un-grounded source citation")
        if semantic:
            goal_words = set(re.findall(r"\w+", p["goal"]["instruction"].casefold()))
            edited_words = set(re.findall(r"\w+", span["text"].casefold()))
            if covering[0]["kind"] == "goal" or goal_words & edited_words:
                raise Invalid("legitimate_goal_word_or_requirement")
        if any(span["text"].casefold() in m["content"].casefold() for m in p["history"]):
            raise Invalid("candidate_present_in_pre_exposure_history")
        result.extend([text[last:start], span["replacement"]])
        last = end
    return "".join(result) + text[last:]


def check_window(victim, snap, edited):
    r = snap["runtime"]
    original_ids, original_full = victim.encode(r["serialized_prompt"], r["generation"])
    prompt = victim.render(r["prefix_messages"], edited)
    ids, full = victim.encode(prompt, r["generation"])
    # Deliberately conservative v1: no edit on any truncated context. This prevents
    # both lost protected text and unrelated text newly entering a shortened window.
    if original_ids != original_full or ids != full:
        raise Invalid("candidate_or_protected_context_truncation")
    if not edited:
        raise Invalid("missing_context")
    return ids
