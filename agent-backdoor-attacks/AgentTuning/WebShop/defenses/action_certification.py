"""
Module 3: Goal-Grounded Action Certification.

This module certifies a proposed action a_t against the goal contract G(q) and
the structured state S_t produced by Module 2.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .goal_contract import GoalContract
from .state_abstraction import StructuredState


_ACTION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\[(.*)\]\s*$", re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\$?\d+(?:\.\d+)?|[A-Z0-9]{10}")
_PRODUCT_ID_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\$?\d+(?:\.\d+)?$")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "please", "show",
    "that", "the", "this", "to", "under", "with", "you", "your",
}

_ACTION_WORDS = {
    "action", "actions", "available", "search", "click", "select", "choose",
    "buy", "purchase", "checkout", "call_tool", "run_command", "query_db",
}

_NEGATION_WORDS = {
    "do", "not", "dont", "don't", "never", "without", "avoid", "exclude",
    "no",
}

_PROGRESS_CONTROL_TERMS = {
    "back", "next", "previous", "prev", "reviews", "review", "description",
    "features", "feature", "size", "color", "colour", "quantity", "select",
    "option", "options", "buy", "now", "purchase", "checkout",
}

_SUPPORTED_OPERATIONS = {
    "search",
    "click",
}

_OPERATION_ALIASES = {
    "buy": {"buy", "purchase", "checkout"},
    "purchase": {"buy", "purchase", "checkout"},
    "checkout": {"buy", "purchase", "checkout"},
    "click": {"click", "select", "choose"},
    "select": {"click", "select", "choose"},
    "search": {"search", "query"},
    "query": {"search", "query"},
}


@dataclass
class ProposedAction:
    """Action representation a_t = (omega_t, rho_t, B_t)."""

    raw_action: str
    operation: Optional[str] = None
    target: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    parse_error: Optional[str] = None

    @property
    def omega(self) -> Optional[str]:
        return self.operation

    @property
    def rho(self) -> Optional[str]:
        return self.target

    @property
    def B(self) -> Dict[str, Any]:
        return self.arguments

    @property
    def valid(self) -> bool:
        return self.operation is not None and self.target is not None and self.parse_error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_action": self.raw_action,
            "omega": self.operation,
            "rho": self.target,
            "B": self.arguments,
            "parse_error": self.parse_error,
        }


@dataclass
class CertificationCheck:
    name: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "evidence": self.evidence,
        }


@dataclass
class ActionCertificationResult:
    proposed_action: ProposedAction
    support: CertificationCheck
    progress: CertificationCheck
    safety: CertificationCheck
    stability: CertificationCheck

    @property
    def accepted(self) -> bool:
        return (
            self.support.passed
            and self.progress.passed
            and self.safety.passed
            and self.stability.passed
        )

    @property
    def z(self) -> int:
        return 1 if self.accepted else 0

    @property
    def accepted_action(self) -> Optional[str]:
        return self.proposed_action.raw_action if self.accepted else None

    @property
    def rejected_action(self) -> Optional[str]:
        return None if self.accepted else self.proposed_action.raw_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": GoalGroundedActionCertification.module_name,
            "z": self.z,
            "accepted": self.accepted,
            "accepted_action": self.accepted_action,
            "rejected_action": self.rejected_action,
            "proposed_action": self.proposed_action.to_dict(),
            "Supp": self.support.to_dict(),
            "Prog": self.progress.to_dict(),
            "Safe": self.safety.to_dict(),
            "Stable": self.stability.to_dict(),
        }


@dataclass
class _StateContext:
    raw_text: str
    terms: Set[str]
    has_search_bar: Optional[bool]
    click_targets: Set[str]
    strings: List[str]


class GoalGroundedActionCertification:
    """Implementation of f_cert(a_t, S_t, G(q)) for WebShop actions."""

    module_name = "goal_grounded_action_certification"

    def certify(
        self,
        action_text: str,
        structured_state: StructuredState,
        goal_contract: GoalContract,
        neutralized_state: Optional[StructuredState] = None,
    ) -> ActionCertificationResult:
        action = self.parse_action(action_text)
        context = self._state_context(structured_state)
        goal_terms = self._expanded_terms(
            self._significant_terms([
                goal_contract.intent,
                goal_contract.positive_constraints,
            ])
        )

        support = self._check_support(action, context, goal_terms)
        progress = self._check_progress(action, context, goal_terms)
        safety = self._check_safety(action, context, goal_contract)
        neutralized_context = self._state_context(
            neutralized_state or structured_state
        )
        neutralized_support = self._check_support(
            action,
            neutralized_context,
            goal_terms,
        )
        neutralized_progress = self._check_progress(
            action,
            neutralized_context,
            goal_terms,
        )
        stability_reasons = [
            "neutralized support: {}".format(reason)
            for reason in neutralized_support.reasons
        ] + [
            "neutralized progress: {}".format(reason)
            for reason in neutralized_progress.reasons
        ]
        stability = CertificationCheck(
            name="Stable",
            passed=(
                neutralized_support.passed
                and neutralized_progress.passed
            ),
            reasons=stability_reasons,
            evidence={
                "neutralized_support": neutralized_support.to_dict(),
                "neutralized_progress": neutralized_progress.to_dict(),
            },
        )

        return ActionCertificationResult(
            proposed_action=action,
            support=support,
            progress=progress,
            safety=safety,
            stability=stability,
        )

    def parse_action(self, action_text: str) -> ProposedAction:
        raw_action = action_text or ""
        match = _ACTION_RE.match(raw_action)
        if not match:
            return ProposedAction(
                raw_action=raw_action,
                parse_error="action is not in operation[target] format",
            )

        operation = match.group(1).strip().lower()
        target = match.group(2).strip()
        arguments: Dict[str, Any] = {"target": target}

        if operation == "search":
            arguments["query"] = target
            arguments["terms"] = sorted(self._significant_terms(target, keep_action_words=True))
        elif operation == "click":
            arguments["click_value"] = target
            arguments["terms"] = sorted(self._significant_terms(target, keep_action_words=True))

        return ProposedAction(
            raw_action=raw_action,
            operation=operation,
            target=target,
            arguments=arguments,
        )

    def _check_support(
        self,
        action: ProposedAction,
        context: _StateContext,
        goal_terms: Set[str],
    ) -> CertificationCheck:
        reasons: List[str] = []
        evidence: Dict[str, Any] = {}

        if not action.valid:
            return CertificationCheck(
                name="Supp",
                passed=False,
                reasons=[action.parse_error or "invalid action"],
            )

        assert action.operation is not None
        assert action.target is not None

        if action.operation not in _SUPPORTED_OPERATIONS:
            return CertificationCheck(
                name="Supp",
                passed=False,
                reasons=[f"unsupported operation: {action.operation}"],
                evidence={"supported_operations": sorted(_SUPPORTED_OPERATIONS)},
            )

        action_terms = self._expanded_terms(self._significant_terms(action.target))
        grounded_terms = goal_terms | context.terms | set(_PROGRESS_CONTROL_TERMS)
        unsupported_terms = sorted(
            term
            for term in action_terms
            if term not in grounded_terms
            and not _PRODUCT_ID_RE.match(term)
            and not _NUMBER_RE.match(term)
        )

        if unsupported_terms:
            reasons.append("action arguments introduce terms not grounded in C+ or S_t")
            evidence["unsupported_arguments"] = unsupported_terms

        if action.operation == "search":
            if context.has_search_bar is False:
                reasons.append("search operation is not available in the current state")
            if action_terms & goal_terms:
                evidence["goal_terms_in_search"] = sorted(action_terms & goal_terms)
            else:
                reasons.append("search target has no overlap with the goal intent or C+")

        if action.operation == "click":
            target_norm = self._normalize(action.target)
            click_grounded = (
                target_norm in context.click_targets
                or target_norm in self._normalize(context.raw_text)
            )
            if not click_grounded:
                reasons.append("click target is not grounded in available state elements")
            evidence["click_target_grounded"] = click_grounded

        return CertificationCheck(
            name="Supp",
            passed=not reasons,
            reasons=reasons or ["operation, target, and arguments are state/goal grounded"],
            evidence=evidence,
        )

    def _check_progress(
        self,
        action: ProposedAction,
        context: _StateContext,
        goal_terms: Set[str],
    ) -> CertificationCheck:
        reasons: List[str] = []
        evidence: Dict[str, Any] = {}

        if not action.valid:
            return CertificationCheck(
                name="Prog",
                passed=False,
                reasons=[action.parse_error or "invalid action"],
            )

        assert action.operation is not None
        assert action.target is not None

        action_terms = self._expanded_terms(self._significant_terms(action.target))
        target_context_terms = set()
        if action.operation == "click":
            target_context_terms = self._expanded_terms(
                self._target_context_terms(action, context)
            )
        overlap = sorted((action_terms | target_context_terms) & goal_terms)
        control_overlap = sorted(action_terms & set(_PROGRESS_CONTROL_TERMS))

        if action.operation == "search":
            if overlap:
                evidence["goal_overlap"] = overlap
            else:
                reasons.append("search does not include intent/C+ terms")

        elif action.operation == "click":
            if overlap:
                evidence["goal_overlap"] = overlap
            elif control_overlap:
                evidence["progress_control_terms"] = control_overlap
            elif action.target and _PRODUCT_ID_RE.match(action.target):
                reasons.append("clicked product id has no local goal-term evidence")
            else:
                reasons.append("click target does not appear to advance I or C+")

        return CertificationCheck(
            name="Prog",
            passed=not reasons,
            reasons=reasons or ["action can make progress toward I or C+"],
            evidence=evidence,
        )

    def _check_safety(
        self,
        action: ProposedAction,
        context: _StateContext,
        goal_contract: GoalContract,
    ) -> CertificationCheck:
        reasons: List[str] = []
        evidence: Dict[str, Any] = {"violated_constraints": []}

        if not action.valid:
            return CertificationCheck(
                name="Safe",
                passed=False,
                reasons=[action.parse_error or "invalid action"],
            )

        action_terms = self._expanded_terms(self._significant_terms(action.raw_action))
        target_context_terms = self._expanded_terms(self._target_context_terms(action, context))
        combined_terms = action_terms | target_context_terms
        combined_text = self._normalize(" ".join([action.raw_action] + self._target_contexts(action, context)))

        for constraint in goal_contract.negative_constraints:
            constraint_terms = self._significant_terms(constraint, keep_action_words=True)
            if not constraint_terms:
                continue

            constraint_set = self._expanded_terms(constraint_terms)
            operation_terms = constraint_set & set(_OPERATION_ALIASES)
            object_terms = constraint_set - set(_OPERATION_ALIASES)
            constraint_text = self._normalize(constraint)

            violates = False
            if constraint_text and constraint_text in combined_text:
                violates = True
            elif operation_terms and object_terms:
                violates = (
                    self._matches_forbidden_operation(action, operation_terms)
                    and object_terms.issubset(combined_terms)
                )
            elif constraint_set.issubset(combined_terms):
                violates = True

            if violates:
                reasons.append(f"action violates explicit forbidden constraint: {constraint}")
                evidence["violated_constraints"].append(constraint)

        return CertificationCheck(
            name="Safe",
            passed=not reasons,
            reasons=reasons or ["action does not violate C-"],
            evidence=evidence,
        )

    def _matches_forbidden_operation(
        self,
        action: ProposedAction,
        operation_terms: Set[str],
    ) -> bool:
        if not action.operation:
            return False

        action_terms = self._expanded_terms(self._significant_terms(action.raw_action, keep_action_words=True))
        for term in operation_terms:
            aliases = _OPERATION_ALIASES.get(term, {term})
            if action.operation in aliases or action_terms & aliases:
                return True
        return False

    def _state_context(self, structured_state: StructuredState) -> _StateContext:
        raw_text = structured_state.raw_text or ""
        strings: List[str] = []
        for element in structured_state.elements:
            self._collect_strings(element.attributes, strings)

        available_actions = self._parse_available_actions(raw_text)
        has_search_bar = available_actions.get("has_search_bar")
        click_targets = {
            self._normalize(target)
            for target in available_actions.get("clickables", [])
            if self._normalize(target)
        }

        terms = self._expanded_terms(self._significant_terms(strings, keep_action_words=True))
        return _StateContext(
            raw_text=raw_text,
            terms=terms,
            has_search_bar=has_search_bar if isinstance(has_search_bar, bool) else None,
            click_targets=click_targets,
            strings=strings,
        )

    def _parse_available_actions(self, raw_text: str) -> Dict[str, Any]:
        match = re.search(r"Available Actions:\s*(.*)$", raw_text or "", flags=re.I | re.S)
        if not match:
            return {}

        payload = match.group(1).strip()
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            return {}

        if not isinstance(parsed, dict):
            return {}

        clickables = parsed.get("clickables", [])
        if not isinstance(clickables, list):
            clickables = []

        return {
            "has_search_bar": parsed.get("has_search_bar"),
            "clickables": [str(item) for item in clickables],
        }

    def _target_contexts(self, action: ProposedAction, context: _StateContext) -> List[str]:
        if not action.target:
            return []

        target_norm = self._normalize(action.target)
        contexts = [
            text
            for text in context.strings
            if target_norm and target_norm in self._normalize(text)
        ]

        action_terms = set(self._significant_terms(action.raw_action, keep_action_words=True))
        if action_terms & {"buy", "purchase", "checkout"}:
            contexts.append(context.raw_text)

        return contexts

    def _target_context_terms(self, action: ProposedAction, context: _StateContext) -> Set[str]:
        return set(self._significant_terms(self._target_contexts(action, context), keep_action_words=True))

    def _collect_strings(self, value: Any, out: List[str]) -> None:
        if value is None:
            return
        if isinstance(value, str):
            out.append(value)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                out.append(str(key))
                self._collect_strings(item, out)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._collect_strings(item, out)
            return
        out.append(str(value))

    def _significant_terms(
        self,
        value: Any,
        keep_action_words: bool = False,
    ) -> Set[str]:
        terms: Set[str] = set()

        def add(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, dict):
                for key, val in item.items():
                    add(key)
                    add(val)
                return
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub_item in item:
                    add(sub_item)
                return

            for token in _TOKEN_RE.findall(str(item).lower()):
                token = token.strip("'_- ")
                if not token:
                    continue
                if token in _STOPWORDS or token in _NEGATION_WORDS:
                    continue
                if not keep_action_words and token in _ACTION_WORDS:
                    continue
                terms.add(token)

        add(value)
        return terms

    def _expanded_terms(self, terms: Iterable[str]) -> Set[str]:
        expanded: Set[str] = set()
        for term in terms:
            expanded.update(self._stem_variants(term.lower()))
        return {term for term in expanded if term}

    @staticmethod
    def _stem_variants(term: str) -> Set[str]:
        variants = {term}
        if not re.search(r"[a-z]", term):
            return variants
        if term.endswith("s") and not term.endswith(("as", "is", "us", "ss")) and len(term) > 3:
            variants.add(term[:-1])
        if not term.endswith("s") and len(term) > 2:
            variants.add(term + "s")
        if term.endswith("ies") and len(term) > 4:
            variants.add(term[:-3] + "y")
        if term.endswith("y") and len(term) > 3:
            variants.add(term[:-1] + "ies")
        return variants

    @staticmethod
    def _normalize(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())
