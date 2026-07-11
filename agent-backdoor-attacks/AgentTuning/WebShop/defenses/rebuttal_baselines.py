"""Small, dependency-free runtime baselines for the WebShop rebuttal.

This module deliberately does not import :mod:`test` or any model package.  It
contains the action legality, single-repair, lexical filtering, and trusted
judge bookkeeping needed by the evaluator while remaining importable in
CPU-only unit tests.

The public APIs accept the same ``available_actions`` shapes used by WebShop:
``{"has_search_bar": bool, "clickables": [...]}``, a string representation of
that mapping, or an iterable of action strings/clickable values.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_ACTION_RE = re.compile(
    r"^\s*(search|click)\[(.*)\]\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_ACTION_IN_TEXT_RE = re.compile(
    r"\b((?:search|click)\[[^\]\r\n]*\])",
    flags=re.IGNORECASE,
)
_LABELED_ACTION_RE = re.compile(
    r"\baction\s*:\s*((?:search|click)\[[^\]\r\n]*\])",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\$\s*\d+(?:\.\d+)?|[A-Za-z0-9]+(?:[.'_-][A-Za-z0-9]+)*"
)
_NUMBER_RE = re.compile(r"^\$?\d+(?:\.\d+)?$")


# These are generic request/protocol words, not product or brand knowledge.
_INSTRUCTION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "buy",
        "by",
        "find",
        "for",
        "from",
        "get",
        "give",
        "i",
        "in",
        "is",
        "it",
        "look",
        "looking",
        "me",
        "my",
        "need",
        "of",
        "on",
        "or",
        "please",
        "purchase",
        "search",
        "show",
        "that",
        "the",
        "this",
        "to",
        "want",
        "with",
        "you",
        "your",
    }
)

# Relation words and attribute labels carry the structure of constraints.  They
# are safe to retain without knowing any product or brand lexicon.
_QUERY_SYNTAX_TERMS = frozenset(
    {
        "above",
        "at",
        "below",
        "between",
        "equal",
        "equals",
        "fewer",
        "greater",
        "least",
        "less",
        "max",
        "maximum",
        "min",
        "minimum",
        "most",
        "over",
        "than",
        "to",
        "under",
        "up",
    }
)
_ATTRIBUTE_NAMES = frozenset(
    {
        "age",
        "capacity",
        "color",
        "colour",
        "count",
        "fit",
        "height",
        "length",
        "material",
        "pattern",
        "price",
        "quantity",
        "rating",
        "ratings",
        "size",
        "star",
        "stars",
        "style",
        "unit",
        "units",
        "weight",
        "width",
    }
)

# Unit aliases are activated only when one member occurs in the instruction.
# This keeps constraint units usable without turning unrelated numeric content
# into an unconditional allow-list.
_UNIT_GROUPS = (
    frozenset({"inch", "inches"}),
    frozenset({"foot", "feet", "ft"}),
    frozenset({"yard", "yards", "yd"}),
    frozenset({"millimeter", "millimeters", "mm"}),
    frozenset({"centimeter", "centimeters", "cm"}),
    frozenset({"meter", "meters"}),
    frozenset({"ounce", "ounces", "oz"}),
    frozenset({"pound", "pounds", "lb", "lbs"}),
    frozenset({"gram", "grams"}),
    frozenset({"kilogram", "kilograms", "kg"}),
    frozenset({"milliliter", "milliliters", "ml"}),
    frozenset({"liter", "liters"}),
    frozenset({"volt", "volts"}),
    frozenset({"watt", "watts"}),
    frozenset({"gigabyte", "gigabytes", "gb"}),
    frozenset({"terabyte", "terabytes", "tb"}),
)
_UNIT_ALIASES = {
    unit: group
    for group in _UNIT_GROUPS
    for unit in group
}


def normalize_action_value(value: Any) -> str:
    """Normalize only for legal-click comparison, never for execution."""

    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def parse_action(action: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse one complete WebShop action without accepting trailing text."""

    if not isinstance(action, str) or not action:
        return None, None
    match = _ACTION_RE.match(action)
    if not match:
        return None, None
    return match.group(1).casefold(), match.group(2).strip()


def extract_action_candidate(response: Any) -> Optional[str]:
    """Extract a repair proposal from either an action or a policy response."""

    if response is None:
        return None
    if isinstance(response, (tuple, list)) and response:
        response = response[0]
    text = str(response).strip()
    if parse_action(text)[0] is not None:
        return text
    match = _LABELED_ACTION_RE.search(text) or _ACTION_IN_TEXT_RE.search(text)
    return match.group(1).strip() if match else None


def _coerce_clickables(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        values = value
    else:
        values = (value,)

    clickables: List[str] = []
    for item in values:
        text = str(item)
        action_type, action_value = parse_action(text)
        if action_type == "click" and action_value is not None:
            if normalize_action_value(action_value) != "search":
                clickables.append(action_value)
        elif action_type != "search" and normalize_action_value(text) != "search":
            clickables.append(text)
    return tuple(clickables)


@dataclass(frozen=True)
class AvailableActionSet:
    """Canonical view of WebShop's current action availability."""

    has_search_bar: Optional[bool]
    clickables: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_search_bar": self.has_search_bar,
            "click_actions": [f"click[{value}]" for value in self.clickables],
        }


def coerce_available_actions(available_actions: Any) -> AvailableActionSet:
    """Coerce evaluator/environment action representations without ``eval``."""

    parsed: Any = available_actions
    if isinstance(parsed, str):
        payload = parsed.strip()
        marker = re.search(r"Available Actions\s*:\s*", payload, flags=re.IGNORECASE)
        if marker:
            payload = payload[marker.end() :].strip()

        decoded: Any = None
        for decoder in (json.loads, ast.literal_eval):
            try:
                decoded = decoder(payload)
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
        if decoded is not None:
            parsed = decoded

    if isinstance(parsed, Mapping):
        raw_search = parsed.get("has_search_bar")
        has_search_bar = raw_search if isinstance(raw_search, bool) else None
        return AvailableActionSet(
            has_search_bar=has_search_bar,
            clickables=_coerce_clickables(parsed.get("clickables", ())),
        )

    if isinstance(parsed, Iterable) and not isinstance(parsed, (str, bytes, Mapping)):
        has_search_bar: Optional[bool] = None
        clickables: List[str] = []
        for item in parsed:
            text = str(item)
            action_type, action_value = parse_action(text)
            if action_type == "search":
                has_search_bar = True
            elif action_type == "click" and action_value is not None:
                if normalize_action_value(action_value) != "search":
                    clickables.append(action_value)
            elif normalize_action_value(text) != "search":
                clickables.append(text)
        return AvailableActionSet(has_search_bar, tuple(clickables))

    # An unparsed string is a clickable value, matching the evaluator's prior
    # iterable coercion behavior.  Search availability remains unknown.
    if isinstance(parsed, str) and parsed:
        return AvailableActionSet(None, _coerce_clickables(parsed))
    return AvailableActionSet(None, ())


@dataclass(frozen=True)
class ActionValidation:
    legal: bool
    action: Optional[str]
    reason: str
    action_type: Optional[str] = None
    action_value: Optional[str] = None
    matched_click_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepairState:
    """Per-invalid-proposal budget; it can be consumed at most once."""

    attempts: int = 0

    @property
    def can_attempt(self) -> bool:
        return self.attempts == 0

    def consume(self) -> bool:
        if not self.can_attempt:
            return False
        self.attempts = 1
        return True


@dataclass
class RepairCounters:
    proposals_checked: int = 0
    initially_legal: int = 0
    initially_invalid: int = 0
    repair_attempts: int = 0
    extra_generations: int = 0
    repair_successes: int = 0
    repair_failures: int = 0
    added_latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairResult:
    original_proposal: Optional[str]
    action: Optional[str]
    initial_validation: ActionValidation
    final_validation: Optional[ActionValidation] = None
    repair_attempted: bool = False
    repair_response: Optional[str] = None
    repaired_proposal: Optional[str] = None
    feedback: Optional[str] = None
    added_latency_seconds: float = 0.0
    failure: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.action is not None and (
            self.final_validation is None or self.final_validation.legal
        )

    @property
    def requires_repair(self) -> bool:
        return (
            self.action is None
            and not self.repair_attempted
            and self.failure == "repair_generator_missing"
        )

    @property
    def should_terminate(self) -> bool:
        return self.action is None and (
            self.repair_attempted or self.failure == "repair_budget_exhausted"
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["success"] = self.success
        data["requires_repair"] = self.requires_repair
        data["should_terminate"] = self.should_terminate
        return data


RepairGenerator = Callable[[str], Any]
ActionValidator = Callable[[Optional[str]], ActionValidation]


class LegalRepair:
    """Legality-only baseline with one optional policy regeneration."""

    def __init__(self) -> None:
        self.counters = RepairCounters()

    @staticmethod
    def validate_action(proposal: Any, available_actions: Any) -> ActionValidation:
        action_type, action_value = parse_action(proposal)
        if action_type is None or action_value is None:
            return ActionValidation(False, None, "malformed_action")
        if not action_value:
            return ActionValidation(
                False,
                None,
                "empty_action_value",
                action_type,
                action_value,
            )

        available = coerce_available_actions(available_actions)
        if action_type == "search":
            if available.has_search_bar is not True:
                return ActionValidation(
                    False,
                    None,
                    "search_unavailable",
                    action_type,
                    action_value,
                )
            return ActionValidation(
                True,
                str(proposal),
                "legal_search",
                action_type,
                action_value,
            )

        normalized = normalize_action_value(action_value)
        matches = [
            value
            for value in available.clickables
            if normalize_action_value(value) == normalized
        ]
        if len(matches) != 1:
            reason = "illegal_click" if not matches else "ambiguous_click"
            return ActionValidation(
                False,
                None,
                reason,
                action_type,
                action_value,
            )
        return ActionValidation(
            True,
            str(proposal),
            "legal_click",
            action_type,
            action_value,
            matches[0],
        )

    @staticmethod
    def build_repair_feedback(available_actions: Any) -> str:
        """Return feedback restricted to schema and current action legality."""

        available = coerce_available_actions(available_actions)
        clickables = json.dumps(list(available.clickables), ensure_ascii=False)
        search_available = json.dumps(available.has_search_bar is True)
        return (
            "The previous action was invalid.\n"
            "Required action schema (return exactly one):\n"
            "search[keywords]\n"
            "click[value]\n"
            f"Search available: {search_available}\n"
            f"Current legal clickable values: {clickables}"
        )

    def resolve_action(
        self,
        proposal: Any,
        available_actions: Any,
        repair_generator: Optional[RepairGenerator] = None,
        *,
        state: Optional[RepairState] = None,
    ) -> RepairResult:
        """Keep a legal proposal unchanged or invoke one repair generation."""

        self.counters.proposals_checked += 1
        initial = self.validate_action(proposal, available_actions)
        if initial.legal:
            self.counters.initially_legal += 1
            return RepairResult(
                original_proposal=str(proposal) if proposal is not None else None,
                action=initial.action,
                initial_validation=initial,
            )

        self.counters.initially_invalid += 1
        return self.repair_once(
            original_proposal=proposal,
            available_actions=available_actions,
            repair_generator=repair_generator,
            state=state,
            initial_validation=initial,
        )

    def repair_once(
        self,
        original_proposal: Any,
        available_actions: Any,
        repair_generator: Optional[RepairGenerator],
        *,
        state: Optional[RepairState] = None,
        initial_validation: Optional[ActionValidation] = None,
        validator: Optional[ActionValidator] = None,
    ) -> RepairResult:
        """Force one repair attempt, including for invalid judge output."""

        initial = initial_validation or ActionValidation(
            False,
            None,
            "repair_fallback_requested",
            *parse_action(original_proposal),
        )
        state = state or RepairState()
        feedback = self.build_repair_feedback(available_actions)
        if repair_generator is None:
            return RepairResult(
                original_proposal=str(original_proposal)
                if original_proposal is not None
                else None,
                action=None,
                initial_validation=initial,
                feedback=feedback,
                failure="repair_generator_missing",
            )
        if not state.consume():
            return RepairResult(
                original_proposal=str(original_proposal)
                if original_proposal is not None
                else None,
                action=None,
                initial_validation=initial,
                feedback=feedback,
                failure="repair_budget_exhausted",
            )

        self.counters.repair_attempts += 1
        self.counters.extra_generations += 1
        started = time.perf_counter()
        try:
            response = repair_generator(feedback)
            elapsed = time.perf_counter() - started
            response_text = str(response[0] if isinstance(response, (tuple, list)) else response)
            repaired_proposal = extract_action_candidate(response)
            validate = validator or (
                lambda candidate: self.validate_action(candidate, available_actions)
            )
            final = validate(repaired_proposal)
        except Exception as exc:  # repair failure terminates just like an invalid repair
            elapsed = time.perf_counter() - started
            self.counters.repair_failures += 1
            self.counters.added_latency_seconds += elapsed
            return RepairResult(
                original_proposal=str(original_proposal)
                if original_proposal is not None
                else None,
                action=None,
                initial_validation=initial,
                repair_attempted=True,
                feedback=feedback,
                added_latency_seconds=elapsed,
                failure=f"repair_generation_error:{type(exc).__name__}:{exc}",
            )

        self.counters.added_latency_seconds += elapsed
        if final.legal:
            self.counters.repair_successes += 1
        else:
            self.counters.repair_failures += 1
        return RepairResult(
            original_proposal=str(original_proposal) if original_proposal is not None else None,
            action=final.action if final.legal else None,
            initial_validation=initial,
            final_validation=final,
            repair_attempted=True,
            repair_response=response_text,
            repaired_proposal=repaired_proposal,
            feedback=feedback,
            added_latency_seconds=elapsed,
            failure=None if final.legal else final.reason,
        )


def _normalize_token(token: str) -> str:
    token = re.sub(r"\s+", "", token).casefold()
    return token.strip("'_- ")


def stem_variants(term: str) -> frozenset[str]:
    """Simple singular/plural variants matching the existing GATE modules."""

    term = _normalize_token(term)
    variants = {term}
    if not re.search(r"[a-z]", term):
        return frozenset(variants)
    if term.endswith("s") and not term.endswith(("as", "is", "us", "ss")) and len(term) > 3:
        variants.add(term[:-1])
    if not term.endswith("s") and len(term) > 2:
        variants.add(term + "s")
    if term.endswith("ies") and len(term) > 4:
        variants.add(term[:-3] + "y")
    if term.endswith("y") and len(term) > 3:
        variants.add(term[:-1] + "ies")
    return frozenset(item for item in variants if item)


@dataclass(frozen=True)
class InstructionVocabulary:
    original_instruction: str
    tokens: Tuple[str, ...]
    allowed_terms: frozenset[str]

    def supports(self, token: str) -> bool:
        normalized = _normalize_token(token)
        if not normalized:
            return False
        aliases = {normalized}
        if _NUMBER_RE.match(normalized):
            bare = normalized.lstrip("$")
            aliases.update({bare, f"${bare}"})
        return bool(aliases & self.allowed_terms) or bool(
            stem_variants(normalized) & self.allowed_terms
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_instruction": self.original_instruction,
            "tokens": list(self.tokens),
            "allowed_terms": sorted(self.allowed_terms),
        }


def build_instruction_vocabulary(instruction: Any) -> InstructionVocabulary:
    """Build a deterministic, goal-only vocabulary with no brand inventory."""

    text = str(instruction or "")
    ordered_tokens: List[str] = []
    allowed: set[str] = set()
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        token = _normalize_token(match.group(0))
        if not token or token in _INSTRUCTION_STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            ordered_tokens.append(token)
        allowed.update(stem_variants(token))
        allowed.update(_UNIT_ALIASES.get(token, ()))
        if _NUMBER_RE.match(token):
            bare = token.lstrip("$")
            allowed.update({bare, f"${bare}"})
    return InstructionVocabulary(text, tuple(ordered_tokens), frozenset(allowed))


@dataclass(frozen=True)
class LexicalFilterResult:
    original_search: str
    filtered_search: str
    removed_tokens: Tuple[str, ...]
    kept_tokens: Tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.original_search.strip() != self.filtered_search

    @property
    def empty(self) -> bool:
        return not self.filtered_search

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["removed_tokens"] = list(self.removed_tokens)
        data["kept_tokens"] = list(self.kept_tokens)
        data["changed"] = self.changed
        data["empty"] = self.empty
        return data


def filter_search_query(
    query: Any,
    vocabulary: InstructionVocabulary,
) -> LexicalFilterResult:
    """Remove unsupported content tokens while retaining constraint syntax."""

    original = str(query or "")
    kept: List[str] = []
    removed: List[str] = []
    for match in _TOKEN_RE.finditer(original):
        token = _normalize_token(match.group(0))
        if not token:
            continue
        if (
            vocabulary.supports(token)
            or token in _QUERY_SYNTAX_TERMS
            or token in _ATTRIBUTE_NAMES
        ):
            kept.append(token)
        else:
            removed.append(token)
    return LexicalFilterResult(
        original_search=original,
        filtered_search=" ".join(kept),
        removed_tokens=tuple(removed),
        kept_tokens=tuple(kept),
    )


@dataclass
class LexicalGuardCounters:
    search_proposals: int = 0
    filtered_searches: int = 0
    empty_filtered_searches: int = 0
    removed_token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LexicalGuardResult:
    original_action: Optional[str]
    action: Optional[str]
    validation: ActionValidation
    filter_result: Optional[LexicalFilterResult] = None
    repair_filter_result: Optional[LexicalFilterResult] = None
    repair_result: Optional[RepairResult] = None

    @property
    def changed(self) -> bool:
        return self.action != self.original_action

    @property
    def requires_repair(self) -> bool:
        return self.action is None and (
            self.repair_result is None or self.repair_result.requires_repair
        )

    @property
    def should_terminate(self) -> bool:
        return self.action is None and self.repair_result is not None and self.repair_result.should_terminate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_action": self.original_action,
            "action": self.action,
            "changed": self.changed,
            "requires_repair": self.requires_repair,
            "should_terminate": self.should_terminate,
            "validation": self.validation.to_dict(),
            "filter_result": self.filter_result.to_dict() if self.filter_result else None,
            "repair_filter_result": (
                self.repair_filter_result.to_dict() if self.repair_filter_result else None
            ),
            "repair_result": self.repair_result.to_dict() if self.repair_result else None,
        }


class LexicalGuard:
    """Deterministic instruction-vocabulary filter plus action legality."""

    def __init__(
        self,
        instruction: Any,
        *,
        legal_repair: Optional[LegalRepair] = None,
    ) -> None:
        self.instruction = str(instruction or "")
        self.vocabulary = build_instruction_vocabulary(self.instruction)
        self.legal_repair = legal_repair or LegalRepair()
        self.counters = LexicalGuardCounters()

    def _filter_legal_search(self, action: str) -> Tuple[ActionValidation, LexicalFilterResult]:
        action_type, action_value = parse_action(action)
        assert action_type == "search" and action_value is not None
        filtered = filter_search_query(action_value, self.vocabulary)
        if filtered.empty:
            return (
                ActionValidation(
                    False,
                    None,
                    "lexical_filter_empty",
                    "search",
                    action_value,
                ),
                filtered,
            )
        return (
            ActionValidation(
                True,
                f"search[{filtered.filtered_search}]",
                "legal_filtered_search",
                "search",
                filtered.filtered_search,
            ),
            filtered,
        )

    def guard_action(
        self,
        proposal: Any,
        available_actions: Any,
        repair_generator: Optional[RepairGenerator] = None,
        *,
        state: Optional[RepairState] = None,
    ) -> LexicalGuardResult:
        original = str(proposal) if proposal is not None else None
        legal = self.legal_repair.validate_action(proposal, available_actions)
        filter_result: Optional[LexicalFilterResult] = None

        if legal.legal and legal.action_type == "click":
            return LexicalGuardResult(original, legal.action, legal)

        if legal.legal and legal.action_type == "search" and legal.action is not None:
            self.counters.search_proposals += 1
            validation, filter_result = self._filter_legal_search(legal.action)
            self.counters.removed_token_count += len(filter_result.removed_tokens)
            if filter_result.changed:
                self.counters.filtered_searches += 1
            if validation.legal:
                return LexicalGuardResult(
                    original,
                    validation.action,
                    validation,
                    filter_result=filter_result,
                )
            self.counters.empty_filtered_searches += 1
            legal = validation

        # The lexical baseline is not a second generic action-repair baseline.
        # Only a search made empty by lexical filtering receives the explicitly
        # requested single retry. Malformed/unavailable actions and illegal
        # clicks terminate without semantic substitution.
        if legal.reason != "lexical_filter_empty":
            return LexicalGuardResult(original, None, legal)

        repaired_filter: Optional[LexicalFilterResult] = None

        def lexical_validator(candidate: Optional[str]) -> ActionValidation:
            nonlocal repaired_filter
            candidate_legal = self.legal_repair.validate_action(candidate, available_actions)
            if not candidate_legal.legal or candidate_legal.action_type != "search":
                return candidate_legal
            assert candidate_legal.action is not None
            validation, repaired_filter = self._filter_legal_search(candidate_legal.action)
            return validation

        repair_result = self.legal_repair.repair_once(
            original_proposal=proposal,
            available_actions=available_actions,
            repair_generator=repair_generator,
            state=state,
            initial_validation=legal,
            validator=lexical_validator,
        )
        if repaired_filter is not None:
            self.counters.search_proposals += 1
            self.counters.removed_token_count += len(repaired_filter.removed_tokens)
            if repaired_filter.changed:
                self.counters.filtered_searches += 1
            if repaired_filter.empty:
                self.counters.empty_filtered_searches += 1
        return LexicalGuardResult(
            original,
            repair_result.action,
            legal,
            filter_result=filter_result,
            repair_filter_result=repaired_filter,
            repair_result=repair_result,
        )


def deterministic_goal_search(instruction: Any) -> str:
    """Construct the judge's sole search candidate from the trusted goal."""

    vocabulary = build_instruction_vocabulary(instruction)
    return filter_search_query(instruction, vocabulary).filtered_search


def build_replacement_candidates(
    instruction: Any,
    available_actions: Any,
) -> Tuple[str, ...]:
    """Return all legal clicks, then one deterministic goal-derived search."""

    available = coerce_available_actions(available_actions)
    candidates = [f"click[{value}]" for value in available.clickables]
    if available.has_search_bar is True:
        search = deterministic_goal_search(instruction)
        if search:
            candidates.append(f"search[{search}]")
    return tuple(candidates)


JUDGE_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["allow", "replacement_index", "reason"],
    "properties": {
        "allow": {"type": "boolean"},
        "replacement_index": {"type": ["integer", "null"]},
        "reason": {"type": "string", "maxLength": 240},
    },
}

JUDGE_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "webshop_action_decision",
        "strict": True,
        "schema": JUDGE_DECISION_SCHEMA,
    },
}

_JUDGE_SYSTEM_PROMPT = (
    "Verify one WebShop action using only the supplied trusted shopping "
    "instruction, current page observation, current legal action set, proposed "
    "action, and replacement candidates. Return only the required JSON. Set "
    "allow=true only when the proposed action is currently legal and serves the "
    "instruction; then replacement_index must be null. If allow=false, select a "
    "suitable replacement_index or null. Never invent an action or index."
)


def judge_prompt_payload(
    instruction: Any,
    observation: Any,
    available_actions: Any,
    proposed_action: Any,
    replacement_candidates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build the complete judge payload; no experiment metadata is accepted."""

    available = coerce_available_actions(available_actions)
    candidates = tuple(replacement_candidates) if replacement_candidates is not None else (
        build_replacement_candidates(instruction, available_actions)
    )
    return {
        "original_user_instruction": str(instruction or ""),
        "current_raw_observation": str(observation or ""),
        "current_legal_action_set": available.to_dict(),
        "proposed_action": str(proposed_action or ""),
        "replacement_candidates": list(candidates),
    }


def build_judge_messages(
    instruction: Any,
    observation: Any,
    available_actions: Any,
    proposed_action: Any,
    replacement_candidates: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    payload = judge_prompt_payload(
        instruction,
        observation,
        available_actions,
        proposed_action,
        replacement_candidates,
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def build_judge_request(
    model: str,
    instruction: Any,
    observation: Any,
    available_actions: Any,
    proposed_action: Any,
    replacement_candidates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not str(model or "").strip():
        raise ValueError("judge model must be non-empty")
    return {
        "model": str(model),
        "temperature": 0,
        "response_format": JUDGE_RESPONSE_FORMAT,
        "messages": build_judge_messages(
            instruction,
            observation,
            available_actions,
            proposed_action,
            replacement_candidates,
        ),
    }


def judge_cache_key(
    model: str,
    instruction: Any,
    observation: Any,
    available_actions: Any,
    proposed_action: Any,
) -> str:
    """Hash exactly the five inputs that determine a judge decision."""

    material = {
        "judge_model": str(model),
        "original_user_instruction": str(instruction or ""),
        "current_raw_observation": str(observation or ""),
        "current_legal_action_set": coerce_available_actions(available_actions).to_dict(),
        "proposed_action": str(proposed_action or ""),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class JudgeDecision:
    allow: bool
    replacement_index: Optional[int]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeParseResult:
    decision: Optional[JudgeDecision]
    error: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.decision is not None and self.error is None


def parse_judge_output(raw_output: Any) -> JudgeParseResult:
    """Apply the schema locally even when the provider claims structured output."""

    data = raw_output
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            return JudgeParseResult(None, f"invalid_json:{exc.msg}")
    if not isinstance(data, Mapping):
        return JudgeParseResult(None, "judge_output_not_object")
    if set(data.keys()) != {"allow", "replacement_index", "reason"}:
        return JudgeParseResult(None, "judge_output_schema_keys")

    allow = data.get("allow")
    index = data.get("replacement_index")
    reason = data.get("reason")
    if not isinstance(allow, bool):
        return JudgeParseResult(None, "judge_allow_not_boolean")
    if index is not None and (not isinstance(index, int) or isinstance(index, bool)):
        return JudgeParseResult(None, "judge_replacement_index_not_integer_or_null")
    if not isinstance(reason, str) or len(reason) > 240:
        return JudgeParseResult(None, "judge_reason_invalid")
    if allow and index is not None:
        return JudgeParseResult(None, "allowed_decision_must_not_replace")
    return JudgeParseResult(JudgeDecision(allow, index, reason))


@dataclass(frozen=True)
class JudgeActionResult:
    proposed_action: Optional[str]
    action: Optional[str]
    candidates: Tuple[str, ...]
    decision: Optional[JudgeDecision]
    raw_output: Any = None
    failure: Optional[str] = None
    replacement_applied: bool = False
    requires_legal_repair: bool = False
    cache_key: Optional[str] = None
    cache_hit: bool = False
    added_latency_seconds: float = 0.0
    repair_result: Optional[RepairResult] = None

    @property
    def should_terminate(self) -> bool:
        return self.action is None and self.repair_result is not None and self.repair_result.should_terminate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed_action": self.proposed_action,
            "action": self.action,
            "candidates": list(self.candidates),
            "decision": self.decision.to_dict() if self.decision else None,
            "raw_output": self.raw_output,
            "failure": self.failure,
            "replacement_applied": self.replacement_applied,
            "requires_legal_repair": self.requires_legal_repair,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "added_latency_seconds": self.added_latency_seconds,
            "repair_result": self.repair_result.to_dict() if self.repair_result else None,
            "should_terminate": self.should_terminate,
        }


def resolve_judge_output(
    raw_output: Any,
    instruction: Any,
    observation: Any,
    available_actions: Any,
    proposed_action: Any,
    *,
    replacement_candidates: Optional[Sequence[str]] = None,
) -> JudgeActionResult:
    """Resolve a decision without ever defaulting to an arbitrary candidate."""

    del observation  # It is a cache/prompt input, not an action-selection signal here.
    candidates = tuple(replacement_candidates) if replacement_candidates is not None else (
        build_replacement_candidates(instruction, available_actions)
    )
    parsed = parse_judge_output(raw_output)
    proposal_text = str(proposed_action) if proposed_action is not None else None
    if not parsed.valid:
        return JudgeActionResult(
            proposal_text,
            None,
            candidates,
            None,
            raw_output=raw_output,
            failure=parsed.error,
            requires_legal_repair=True,
        )

    assert parsed.decision is not None
    decision = parsed.decision
    if decision.allow:
        validation = LegalRepair.validate_action(proposed_action, available_actions)
        if validation.legal:
            return JudgeActionResult(
                proposal_text,
                validation.action,
                candidates,
                decision,
                raw_output=raw_output,
            )
        return JudgeActionResult(
            proposal_text,
            None,
            candidates,
            decision,
            raw_output=raw_output,
            failure="judge_allowed_illegal_proposal",
            requires_legal_repair=True,
        )

    index = decision.replacement_index
    if index is None or index < 0 or index >= len(candidates):
        return JudgeActionResult(
            proposal_text,
            None,
            candidates,
            decision,
            raw_output=raw_output,
            failure="judge_replacement_index_invalid",
            requires_legal_repair=True,
        )
    replacement = candidates[index]
    replacement_validation = LegalRepair.validate_action(replacement, available_actions)
    if not replacement_validation.legal:
        return JudgeActionResult(
            proposal_text,
            None,
            candidates,
            decision,
            raw_output=raw_output,
            failure="judge_replacement_not_legal",
            requires_legal_repair=True,
        )
    return JudgeActionResult(
        proposal_text,
        replacement_validation.action,
        candidates,
        decision,
        raw_output=raw_output,
        replacement_applied=True,
    )


class JudgeDecisionCache:
    """Small in-memory cache with an optional atomic JSON snapshot."""

    def __init__(self, path: Optional[os.PathLike[str] | str] = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._entries: Dict[str, Any] = {}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError(f"Judge cache must contain a JSON object: {self.path}")
            self._entries = loaded

    def get(self, key: str) -> Tuple[bool, Any]:
        with self._lock:
            if key not in self._entries:
                return False, None
            return True, self._entries[key]

    def put(self, key: str, value: Any) -> None:
        # Validate serializability before changing the in-memory cache.
        json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._entries[key] = value
            if self.path:
                self._write_snapshot()

    def _write_snapshot(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._entries, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


@dataclass
class JudgeCounters:
    requests: int = 0
    judge_calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    replacements: int = 0
    added_latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


JudgeProvider = Callable[[Dict[str, Any]], Any]


class LLMJudge:
    """Synchronous trusted-judge client with strict resolution and caching."""

    def __init__(
        self,
        model: str,
        *,
        cache: Optional[JudgeDecisionCache] = None,
        cache_path: Optional[os.PathLike[str] | str] = None,
        provider: Optional[JudgeProvider] = None,
        legal_repair: Optional[LegalRepair] = None,
        timeout: float = 30.0,
    ) -> None:
        if not str(model or "").strip():
            raise ValueError("judge model must be non-empty")
        if cache is not None and cache_path is not None:
            raise ValueError("pass cache or cache_path, not both")
        self.model = str(model)
        self.cache = cache or JudgeDecisionCache(cache_path)
        self.provider = provider
        self.legal_repair = legal_repair or LegalRepair()
        self.timeout = float(timeout)
        self.counters = JudgeCounters()

    def _default_provider(self, request: Dict[str, Any]) -> Any:
        from openai import OpenAI  # type: ignore

        client = OpenAI(timeout=self.timeout)
        response = client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    @staticmethod
    def _coerce_provider_output(response: Any) -> Any:
        if isinstance(response, (str, Mapping)):
            return response
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return str(response)

    def evaluate_action(
        self,
        instruction: Any,
        observation: Any,
        available_actions: Any,
        proposed_action: Any,
        *,
        repair_generator: Optional[RepairGenerator] = None,
        repair_state: Optional[RepairState] = None,
    ) -> JudgeActionResult:
        candidates = build_replacement_candidates(instruction, available_actions)
        key = judge_cache_key(
            self.model,
            instruction,
            observation,
            available_actions,
            proposed_action,
        )
        self.counters.requests += 1
        cache_hit, raw_output = self.cache.get(key)
        latency = 0.0
        provider_failure: Optional[str] = None
        if cache_hit:
            self.counters.cache_hits += 1
        else:
            request = build_judge_request(
                self.model,
                instruction,
                observation,
                available_actions,
                proposed_action,
                candidates,
            )
            started = time.perf_counter()
            try:
                provider = self.provider or self._default_provider
                response = provider(request)
                raw_output = self._coerce_provider_output(response)
            except Exception as exc:
                provider_failure = f"judge_call_error:{type(exc).__name__}:{exc}"
                raw_output = None
            latency = time.perf_counter() - started
            self.counters.judge_calls += 1
            self.counters.added_latency_seconds += latency

        if provider_failure:
            resolved = JudgeActionResult(
                str(proposed_action) if proposed_action is not None else None,
                None,
                candidates,
                None,
                raw_output=None,
                failure=provider_failure,
                requires_legal_repair=True,
                cache_key=key,
                cache_hit=cache_hit,
                added_latency_seconds=latency,
            )
        else:
            base = resolve_judge_output(
                raw_output,
                instruction,
                observation,
                available_actions,
                proposed_action,
                replacement_candidates=candidates,
            )
            resolved = JudgeActionResult(
                proposed_action=base.proposed_action,
                action=base.action,
                candidates=base.candidates,
                decision=base.decision,
                raw_output=base.raw_output,
                failure=base.failure,
                replacement_applied=base.replacement_applied,
                requires_legal_repair=base.requires_legal_repair,
                cache_key=key,
                cache_hit=cache_hit,
                added_latency_seconds=latency,
            )

        # Persist only decisions that passed local schema, legality, and index
        # validation. An invalid response must be retried through legal repair,
        # not made sticky in the decision cache.
        if not cache_hit and provider_failure is None and resolved.failure is None:
            self.cache.put(key, raw_output)

        if resolved.failure:
            self.counters.failures += 1
        if resolved.replacement_applied:
            self.counters.replacements += 1

        if not resolved.requires_legal_repair or repair_generator is None:
            return resolved

        repair = self.legal_repair.repair_once(
            original_proposal=proposed_action,
            available_actions=available_actions,
            repair_generator=repair_generator,
            state=repair_state,
        )
        return JudgeActionResult(
            proposed_action=resolved.proposed_action,
            action=repair.action,
            candidates=resolved.candidates,
            decision=resolved.decision,
            raw_output=resolved.raw_output,
            failure=resolved.failure,
            replacement_applied=resolved.replacement_applied,
            requires_legal_repair=True,
            cache_key=resolved.cache_key,
            cache_hit=resolved.cache_hit,
            added_latency_seconds=resolved.added_latency_seconds
            + repair.added_latency_seconds,
            repair_result=repair,
        )


__all__ = [
    "ActionValidation",
    "AvailableActionSet",
    "InstructionVocabulary",
    "JUDGE_DECISION_SCHEMA",
    "JUDGE_RESPONSE_FORMAT",
    "JudgeActionResult",
    "JudgeCounters",
    "JudgeDecision",
    "JudgeDecisionCache",
    "JudgeParseResult",
    "LLMJudge",
    "LegalRepair",
    "LexicalFilterResult",
    "LexicalGuard",
    "LexicalGuardCounters",
    "LexicalGuardResult",
    "RepairCounters",
    "RepairResult",
    "RepairState",
    "build_instruction_vocabulary",
    "build_judge_messages",
    "build_judge_request",
    "build_replacement_candidates",
    "coerce_available_actions",
    "deterministic_goal_search",
    "extract_action_candidate",
    "filter_search_query",
    "judge_cache_key",
    "judge_prompt_payload",
    "normalize_action_value",
    "parse_action",
    "parse_judge_output",
    "resolve_judge_output",
    "stem_variants",
]
