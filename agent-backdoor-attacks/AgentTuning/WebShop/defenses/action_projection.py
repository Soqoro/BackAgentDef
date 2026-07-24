"""
Module 4: Goal-Constrained Action Projection.

If Module 3 rejects a proposed action, this module projects it into a legal
WebShop action that is supported by the current state and safe under the goal
contract.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from .action_certification import (
    ActionCertificationResult,
    GoalGroundedActionCertification,
    ProposedAction,
)
from .goal_contract import GoalContract
from .state_abstraction import StructuredState


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\$?\d+(?:\.\d+)?|[A-Z0-9]{10}")
_ACTION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\[(.*)\]\s*$", re.DOTALL)

_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "find", "for",
    "from", "get", "give", "i", "in", "is", "it", "look", "looking",
    "me", "my", "need", "of", "on", "or", "please", "purchase", "search",
    "show", "that", "the", "this", "to", "under", "want", "with", "you",
    "your",
}

_CONTROL_FALLBACKS = (
    "back",
    "previous",
    "prev",
    "reviews",
    "description",
    "features",
    "next",
)


@dataclass
class ProjectionCandidate:
    action: str
    certification: ActionCertificationResult
    progress_score: float
    distance: float
    score: float
    source: str = "candidate"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "progress_score": self.progress_score,
            "distance": self.distance,
            "score": self.score,
            "source": self.source,
            "certification": self.certification.to_dict(),
        }


@dataclass
class ActionProjectionResult:
    original_action: str
    projected_action: str
    original_certification: Optional[ActionCertificationResult] = None
    projected_certification: Optional[ActionCertificationResult] = None
    selected_candidate: Optional[ProjectionCandidate] = None
    candidates: List[ProjectionCandidate] = field(default_factory=list)
    legal_actions: List[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.projected_action != self.original_action

    @property
    def accepted(self) -> bool:
        if self.projected_certification is None:
            return False
        return self.projected_certification.accepted

    def to_dict(self, max_candidates: int = 20) -> Dict[str, Any]:
        candidates = self.candidates[:max_candidates]
        return {
            "module": GoalConstrainedActionProjection.module_name,
            "original_action": self.original_action,
            "projected_action": self.projected_action,
            "changed": self.changed,
            "accepted": self.accepted,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "legal_actions_count": len(self.legal_actions),
            "legal_actions_preview": self.legal_actions[:max_candidates],
            "candidate_count": len(self.candidates),
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "candidates_preview": [candidate.to_dict() for candidate in candidates],
            "original_certification": (
                self.original_certification.to_dict()
                if self.original_certification is not None
                else None
            ),
            "projected_certification": (
                self.projected_certification.to_dict()
                if self.projected_certification is not None
                else None
            ),
        }


class GoalConstrainedActionProjection:
    """Implementation of f_proj(a_t, A_t, S_t, G(q)) for WebShop."""

    module_name = "goal_constrained_action_projection"

    def __init__(
        self,
        certifier: Optional[GoalGroundedActionCertification] = None,
        lambda_progress: float = 1.0,
        lambda_distance: float = 0.25,
    ) -> None:
        self.certifier = certifier or GoalGroundedActionCertification()
        self.lambda_progress = lambda_progress
        self.lambda_distance = lambda_distance

    def project(
        self,
        action_text: str,
        legal_actions: Any,
        structured_state: StructuredState,
        goal_contract: GoalContract,
        certification_result: Optional[ActionCertificationResult] = None,
        neutralized_state: Optional[StructuredState] = None,
    ) -> ActionProjectionResult:
        counterfactual_state = neutralized_state or structured_state
        original_certification = certification_result or self.certifier.certify(
            action_text or "",
            structured_state,
            goal_contract,
            neutralized_state=counterfactual_state,
        )

        if original_certification.accepted:
            return ActionProjectionResult(
                original_action=action_text or "",
                projected_action=action_text or "",
                original_certification=original_certification,
                projected_certification=original_certification,
                legal_actions=self.legal_action_strings(legal_actions, goal_contract, action_text),
            )

        legal_action_strings = self.legal_action_strings(
            legal_actions,
            goal_contract,
            action_text,
        )

        candidates = self._goal_supported_candidates(
            original_action=action_text or "",
            legal_action_strings=legal_action_strings,
            structured_state=structured_state,
            neutralized_state=counterfactual_state,
            goal_contract=goal_contract,
        )

        if candidates:
            selected = max(candidates, key=lambda item: (item.score, -item.distance, item.action))
            return ActionProjectionResult(
                original_action=action_text or "",
                projected_action=selected.action,
                original_certification=original_certification,
                projected_certification=selected.certification,
                selected_candidate=selected,
                candidates=sorted(candidates, key=lambda item: item.score, reverse=True),
                legal_actions=legal_action_strings,
            )

        fallback_action = self._safe_fallback_action(legal_actions, goal_contract, action_text)
        fallback_certification = self.certifier.certify(
            fallback_action,
            structured_state,
            goal_contract,
            neutralized_state=counterfactual_state,
        )

        return ActionProjectionResult(
            original_action=action_text or "",
            projected_action=fallback_action,
            original_certification=original_certification,
            projected_certification=fallback_certification,
            candidates=[],
            legal_actions=legal_action_strings,
            fallback_used=True,
            fallback_reason="no legal action satisfied Supp and Safe",
        )

    def legal_action_strings(
        self,
        legal_actions: Any,
        goal_contract: GoalContract,
        original_action: Optional[str] = None,
    ) -> List[str]:
        action_set: List[str] = []
        parsed = self._coerce_available_actions(legal_actions)
        clickables = [str(item) for item in parsed.get("clickables", [])]
        has_search_bar = parsed.get("has_search_bar")

        if has_search_bar is True:
            goal_query = self._goal_search_query(goal_contract)
            if goal_query:
                action_set.append(f"search[{goal_query}]")

            projected_query = self._projected_search_query(original_action, goal_contract)
            if projected_query:
                action_set.append(f"search[{projected_query}]")

        for clickable in clickables:
            action_set.append(f"click[{clickable}]")

        return self._dedupe_actions(action_set)

    def _goal_supported_candidates(
        self,
        original_action: str,
        legal_action_strings: Sequence[str],
        structured_state: StructuredState,
        neutralized_state: StructuredState,
        goal_contract: GoalContract,
    ) -> List[ProjectionCandidate]:
        candidates: List[ProjectionCandidate] = []
        for candidate_action in legal_action_strings:
            certification = self.certifier.certify(
                candidate_action,
                structured_state,
                goal_contract,
                neutralized_state=neutralized_state,
            )
            # Projection is constrained by the same complete goal certificate
            # as the original action. In particular, a trigger-dependent legal
            # action must not re-enter through the repair path.
            if not certification.accepted:
                continue

            progress_score = self._progress_score(certification)
            distance = self._distance(candidate_action, original_action)
            score = self.lambda_progress * progress_score - self.lambda_distance * distance
            candidates.append(
                ProjectionCandidate(
                    action=candidate_action,
                    certification=certification,
                    progress_score=progress_score,
                    distance=distance,
                    score=score,
                    source="legal_goal_supported_action",
                )
            )
        return candidates

    def _safe_fallback_action(
        self,
        legal_actions: Any,
        goal_contract: GoalContract,
        original_action: Optional[str],
    ) -> str:
        parsed = self._coerce_available_actions(legal_actions)
        clickables = [str(item) for item in parsed.get("clickables", [])]

        for preferred in _CONTROL_FALLBACKS:
            for clickable in clickables:
                if self._normalize(clickable) == preferred:
                    return f"click[{clickable}]"

        if parsed.get("has_search_bar") is True:
            query = self._goal_search_query(goal_contract)
            if query:
                return f"search[{query}]"

        if clickables:
            return f"click[{clickables[0]}]"

        return original_action or "click[back]"

    def _coerce_available_actions(self, legal_actions: Any) -> Dict[str, Any]:
        if isinstance(legal_actions, dict):
            return {
                "has_search_bar": legal_actions.get("has_search_bar"),
                "clickables": self._coerce_clickables(legal_actions.get("clickables", [])),
            }

        if isinstance(legal_actions, str):
            parsed = self._parse_available_actions_string(legal_actions)
            if parsed:
                return parsed
            return {
                "has_search_bar": None,
                "clickables": self._coerce_action_list([legal_actions]),
            }

        if isinstance(legal_actions, Iterable):
            return {
                "has_search_bar": None,
                "clickables": self._coerce_action_list(list(legal_actions)),
            }

        return {"has_search_bar": None, "clickables": []}

    def _parse_available_actions_string(self, text: str) -> Dict[str, Any]:
        payload = text.strip()
        match = re.search(r"Available Actions:\s*(.*)$", payload, flags=re.I | re.S)
        if match:
            payload = match.group(1).strip()

        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            return {}

        if not isinstance(parsed, dict):
            return {}

        return {
            "has_search_bar": parsed.get("has_search_bar"),
            "clickables": self._coerce_clickables(parsed.get("clickables", [])),
        }

    def _coerce_clickables(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(item) for item in value]
        return [str(value)]

    def _coerce_action_list(self, values: Sequence[Any]) -> List[str]:
        clickables = []
        for value in values:
            text = str(value)
            match = _ACTION_RE.match(text)
            if match and match.group(1).lower() == "click":
                clickables.append(match.group(2).strip())
            elif not match:
                clickables.append(text)
        return clickables

    def _goal_search_query(self, goal_contract: GoalContract) -> str:
        return " ".join(self._query_terms([goal_contract.intent, goal_contract.positive_constraints]))

    def _projected_search_query(
        self,
        original_action: Optional[str],
        goal_contract: GoalContract,
    ) -> str:
        if not original_action:
            return ""

        match = _ACTION_RE.match(original_action)
        if not match or match.group(1).lower() != "search":
            return ""

        original_terms = self._query_terms(match.group(2))
        goal_terms = set(self._query_terms([goal_contract.intent, goal_contract.positive_constraints]))
        kept_terms = [term for term in original_terms if term in goal_terms]
        return " ".join(kept_terms)

    def _query_terms(self, value: Any) -> List[str]:
        terms: List[str] = []

        def add(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, dict):
                for key, sub_value in item.items():
                    add(key)
                    add(sub_value)
                return
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub_item in item:
                    add(sub_item)
                return

            for token in _TOKEN_RE.findall(str(item).lower()):
                clean = token.strip("'_- ")
                if not clean or clean in _QUERY_STOPWORDS:
                    continue
                if clean not in terms:
                    terms.append(clean)

        add(value)
        return terms

    def _progress_score(self, certification: ActionCertificationResult) -> float:
        score = 1.0 if certification.progress.passed else 0.0
        progress_evidence = certification.progress.evidence
        support_evidence = certification.support.evidence
        progress_overlap = progress_evidence.get("goal_overlap", [])
        support_overlap = support_evidence.get("goal_terms_in_search", [])
        control = progress_evidence.get("progress_control_terms", [])
        if isinstance(progress_overlap, list):
            score += min(0.4, 0.06 * len(progress_overlap))
        if isinstance(support_overlap, list):
            score += min(0.4, 0.06 * len(support_overlap))
        if isinstance(control, list):
            score += min(0.1, 0.02 * len(control))
        return score

    def _distance(self, candidate_action: str, original_action: str) -> float:
        candidate = self.certifier.parse_action(candidate_action)
        original = self.certifier.parse_action(original_action)

        if not candidate.valid or not original.valid:
            return 1.0 if candidate_action != original_action else 0.0

        operation_distance = 0.0 if candidate.operation == original.operation else 1.0
        candidate_terms = set(self._query_terms(candidate.target))
        original_terms = set(self._query_terms(original.target))

        if candidate_terms or original_terms:
            union = candidate_terms | original_terms
            intersection = candidate_terms & original_terms
            target_distance = 1.0 - (len(intersection) / len(union))
        else:
            target_distance = 0.0 if self._normalize(candidate.target) == self._normalize(original.target) else 1.0

        return operation_distance + target_distance

    def _dedupe_actions(self, actions: Sequence[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for action in actions:
            key = self._normalize(action)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(action)
        return out

    @staticmethod
    def _normalize(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())
