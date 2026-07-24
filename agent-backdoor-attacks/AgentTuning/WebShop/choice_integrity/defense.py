"""Commitment-time choice certification and deterministic legal repair.

This module is intentionally independent of model, WebShop, Lucene, and API
clients.  The existing goal-integrity layer should run first and pass its
decision into :class:`ChoiceIntegrityDefense`.  Preference logic can never
override a failed hard-goal decision.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ledger import (
    CandidateLedger,
    CandidateReference,
    LedgerEntry,
    LedgerIncompleteError,
)
from .schema import Candidate, Preference


CHOICE_INTEGRITY_ABLATIONS = ("full", "no_ledger", "no_dominance")

_ACTION_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\[(.*)\]\s*$",
    flags=re.DOTALL,
)
_COMMIT_VALUE = "buy now"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def parse_action(action: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not action:
        return None, None
    match = _ACTION_RE.match(str(action))
    if not match:
        return None, None
    return match.group(1).lower(), match.group(2).strip()


def is_commitment_action(action: Optional[str]) -> bool:
    operation, target = parse_action(action)
    return operation == "click" and _normalize_text(target) == _COMMIT_VALUE


class DecisionStatus(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"


@dataclass(frozen=True)
class ChoiceDecision:
    status: DecisionStatus
    action: Optional[str]
    reason: str
    selected_candidate_id: Optional[str] = None
    dominating_candidate_ids: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    ablation: str = "full"

    @property
    def accepted(self) -> bool:
        return self.status == DecisionStatus.ACCEPT

    @property
    def blocked(self) -> bool:
        return self.status in (DecisionStatus.REJECT, DecisionStatus.DEFER)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "accepted": self.accepted,
            "action": self.action,
            "reason": self.reason,
            "selected_candidate_id": self.selected_candidate_id,
            "dominating_candidate_ids": list(self.dominating_candidate_ids),
            "missing": list(self.missing),
            "ablation": self.ablation,
        }


class ChoiceIntegrityGuard:
    """Apply hard feasibility and non-dominance only at ``click[Buy Now]``."""

    def __init__(
        self,
        *,
        ablation: str = "full",
        require_evidence: bool = True,
    ) -> None:
        if ablation not in CHOICE_INTEGRITY_ABLATIONS:
            raise ValueError(
                "unknown choice-integrity ablation {!r}; expected one of {}".format(
                    ablation,
                    ", ".join(CHOICE_INTEGRITY_ABLATIONS),
                )
            )
        self.ablation = ablation
        self.require_evidence = bool(require_evidence)

    def certify(
        self,
        action: Optional[str],
        *,
        selected_candidate: Optional[CandidateReference],
        ledger: Optional[CandidateLedger],
        preference: Optional[Preference],
    ) -> ChoiceDecision:
        if not is_commitment_action(action):
            return ChoiceDecision(
                status=DecisionStatus.ACCEPT,
                action=action,
                reason="non_commitment_action",
                ablation=self.ablation,
            )

        selected_id = _reference_id(selected_candidate)

        # Without an explicit order, this layer is exactly a no-op.  Hard-goal
        # feasibility remains the responsibility of the preceding goal guard.
        if preference is None:
            return ChoiceDecision(
                status=DecisionStatus.ACCEPT,
                action=action,
                reason="no_explicit_preference",
                selected_candidate_id=selected_id,
                ablation=self.ablation,
            )

        if selected_candidate is None:
            return ChoiceDecision(
                status=DecisionStatus.DEFER,
                action=action,
                reason="selected_candidate_unknown",
                missing=("selected_candidate",),
                ablation=self.ablation,
            )

        # The no-ledger ablation removes comparison coverage, not the hard
        # feasibility requirement.  A directly supplied Candidate may still
        # prove hard feasibility.  Otherwise we fail closed.
        if self.ablation == "no_ledger":
            feasible = _direct_feasibility(selected_candidate)
            if feasible is False:
                return ChoiceDecision(
                    status=DecisionStatus.REJECT,
                    action=action,
                    reason="hard_infeasible_candidate",
                    selected_candidate_id=selected_id,
                    ablation=self.ablation,
                )
            if feasible is None:
                return ChoiceDecision(
                    status=DecisionStatus.DEFER,
                    action=action,
                    reason="hard_feasibility_unknown_without_ledger",
                    selected_candidate_id=selected_id,
                    missing=("feasible",),
                    ablation=self.ablation,
                )
            return ChoiceDecision(
                status=DecisionStatus.ACCEPT,
                action=action,
                reason="ablation_no_ledger",
                selected_candidate_id=selected_id,
                ablation=self.ablation,
            )

        if ledger is None:
            return ChoiceDecision(
                status=DecisionStatus.DEFER,
                action=action,
                reason="candidate_ledger_unavailable",
                selected_candidate_id=selected_id,
                missing=("ledger",),
                ablation=self.ablation,
            )

        entry = ledger.get(selected_candidate)
        if entry is None:
            return ChoiceDecision(
                status=DecisionStatus.DEFER,
                action=action,
                reason="selected_candidate_absent_from_ledger",
                selected_candidate_id=selected_id,
                missing=("selected_candidate",),
                ablation=self.ablation,
            )
        if "feasible" in entry.ambiguous_fields:
            return ChoiceDecision(
                status=DecisionStatus.DEFER,
                action=action,
                reason="hard_feasibility_conflict",
                selected_candidate_id=entry.candidate_id,
                missing=("feasible:conflict",),
                ablation=self.ablation,
            )
        if entry.feasible is False:
            return ChoiceDecision(
                status=DecisionStatus.REJECT,
                action=action,
                reason="hard_infeasible_candidate",
                selected_candidate_id=entry.candidate_id,
                ablation=self.ablation,
            )
        if entry.feasible is None:
            return ChoiceDecision(
                status=DecisionStatus.DEFER,
                action=action,
                reason="hard_feasibility_unknown",
                selected_candidate_id=entry.candidate_id,
                missing=("feasible",),
                ablation=self.ablation,
            )

        if self.ablation == "no_dominance":
            return ChoiceDecision(
                status=DecisionStatus.ACCEPT,
                action=action,
                reason="ablation_no_dominance",
                selected_candidate_id=entry.candidate_id,
                ablation=self.ablation,
            )

        try:
            dominators = ledger.dominators(
                entry,
                preference,
                require_evidence=self.require_evidence,
            )
        except LedgerIncompleteError as exc:
            flattened_missing = tuple(
                "{}:{}".format(candidate_id, field_name)
                for candidate_id in sorted(exc.missing)
                for field_name in exc.missing[candidate_id]
            )
            return ChoiceDecision(
                status=DecisionStatus.DEFER,
                action=action,
                reason="comparison_evidence_incomplete",
                selected_candidate_id=entry.candidate_id,
                missing=flattened_missing or ("comparison_coverage",),
                ablation=self.ablation,
            )

        if dominators:
            return ChoiceDecision(
                status=DecisionStatus.REJECT,
                action=action,
                reason="selected_candidate_is_dominated",
                selected_candidate_id=entry.candidate_id,
                dominating_candidate_ids=tuple(
                    candidate.candidate_id for candidate in dominators
                ),
                ablation=self.ablation,
            )

        return ChoiceDecision(
            status=DecisionStatus.ACCEPT,
            action=action,
            reason="hard_feasible_and_nondominated",
            selected_candidate_id=entry.candidate_id,
            ablation=self.ablation,
        )


def _reference_id(reference: Optional[CandidateReference]) -> Optional[str]:
    if reference is None:
        return None
    if isinstance(reference, str):
        return reference
    if isinstance(reference, Candidate):
        return reference.candidate_id
    if isinstance(reference, LedgerEntry):
        return reference.candidate_id
    raise TypeError("selected_candidate must be an ID, Candidate, or LedgerEntry")


def _direct_feasibility(reference: CandidateReference) -> Optional[bool]:
    if isinstance(reference, Candidate):
        return reference.feasible
    if isinstance(reference, LedgerEntry):
        if "feasible" in reference.ambiguous_fields:
            return None
        return reference.feasible
    return None


@dataclass(frozen=True)
class AvailableActions:
    has_search_bar: Optional[bool]
    clickables: Tuple[str, ...]

    @classmethod
    def parse(cls, value: Any) -> "AvailableActions":
        if isinstance(value, AvailableActions):
            return value
        if isinstance(value, Mapping):
            search = value.get("has_search_bar")
            search = search if isinstance(search, bool) else None
            return cls(search, _coerce_clickables(value.get("clickables", ())))
        if isinstance(value, str):
            payload = value.strip()
            marker = re.search(r"Available Actions:\s*(.*)$", payload, flags=re.I | re.S)
            if marker:
                payload = marker.group(1).strip()
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, Mapping):
                return cls.parse(parsed)
            if isinstance(parsed, Iterable) and not isinstance(parsed, (str, bytes)):
                return cls(None, _actions_to_clickables(parsed))
            return cls(None, _actions_to_clickables((value,)))
        if isinstance(value, Iterable):
            return cls(None, _actions_to_clickables(value))
        return cls(None, ())

    def matching_clickable(self, value: str) -> Optional[str]:
        normalized = _normalize_text(value)
        for clickable in self.clickables:
            if _normalize_text(clickable) == normalized:
                return clickable
        return None

    def legal_action(self, action: Optional[str]) -> bool:
        operation, target = parse_action(action)
        if operation == "search":
            return self.has_search_bar is True and bool(target and target.strip())
        if operation == "click" and target is not None:
            return self.matching_clickable(target) is not None
        return False


def _coerce_clickables(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _actions_to_clickables(values: Iterable[Any]) -> Tuple[str, ...]:
    result: List[str] = []
    for value in values:
        text = str(value)
        operation, target = parse_action(text)
        if operation == "click" and target is not None:
            result.append(target)
        elif operation is None:
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class PlanResult:
    action: Optional[str]
    reason: str
    target_candidate_id: Optional[str] = None
    legal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "target_candidate_id": self.target_candidate_id,
            "legal": self.legal,
        }


class DeterministicChoicePlanner:
    """Choose a legal next action toward a public-ledger preference winner."""

    def __init__(self, *, require_evidence: bool = True) -> None:
        self.require_evidence = bool(require_evidence)

    def plan(
        self,
        *,
        ledger: Optional[CandidateLedger],
        preference: Optional[Preference],
        available_actions: Any,
        canonical_query: str,
        current_candidate: Optional[CandidateReference] = None,
        selected_options: Optional[Mapping[str, str]] = None,
        current_page: Optional[int] = None,
    ) -> PlanResult:
        available = AvailableActions.parse(available_actions)
        target: Optional[LedgerEntry] = None
        incomplete = False

        if ledger is not None and preference is not None:
            try:
                winners = ledger.winners(
                    preference,
                    require_evidence=self.require_evidence,
                )
                # winner IDs are deterministic and contain no evaluator labels.
                target = min(winners, key=lambda item: item.candidate_id)
            except LedgerIncompleteError:
                incomplete = True

        if target is not None:
            current = _resolve_current_entry(current_candidate, ledger)
            current_asin = current.asin if current is not None else _reference_asin(
                current_candidate
            )

            if current_asin is not None and _normalize_text(current_asin) == _normalize_text(
                target.asin
            ):
                selected = {
                    _normalize_text(name): _normalize_text(value)
                    for name, value in (selected_options or {}).items()
                }
                options_complete = True
                for option_name, option_value in target.options:
                    if selected.get(_normalize_text(option_name)) == _normalize_text(
                        option_value
                    ):
                        continue
                    options_complete = False
                    clickable = available.matching_clickable(option_value)
                    if clickable is not None:
                        action = "click[{}]".format(clickable)
                        return PlanResult(
                            action=action,
                            reason="select_preferred_candidate_option",
                            target_candidate_id=target.candidate_id,
                            legal=True,
                        )

                # A product-option identity is the comparison unit.  Never
                # commit merely because the ASIN matches when a required
                # option is still unselected (or unavailable on this page).
                if options_complete:
                    buy = available.matching_clickable("Buy Now")
                    if buy is not None:
                        return PlanResult(
                            action="click[{}]".format(buy),
                            reason="commit_preferred_candidate",
                            target_candidate_id=target.candidate_id,
                            legal=True,
                        )

            asin_click = available.matching_clickable(target.asin)
            if asin_click is not None:
                return PlanResult(
                    action="click[{}]".format(asin_click),
                    reason="open_preferred_candidate",
                    target_candidate_id=target.candidate_id,
                    legal=True,
                )

            if current_page is not None and target.page is not None:
                if current_page < target.page:
                    next_click = available.matching_clickable("Next >")
                    if next_click is not None:
                        return PlanResult(
                            action="click[{}]".format(next_click),
                            reason="navigate_to_preferred_candidate_page",
                            target_candidate_id=target.candidate_id,
                            legal=True,
                        )
                if current_page > target.page:
                    prev_click = (
                        available.matching_clickable("< Prev")
                        or available.matching_clickable("Previous")
                    )
                    if prev_click is not None:
                        return PlanResult(
                            action="click[{}]".format(prev_click),
                            reason="navigate_to_preferred_candidate_page",
                            target_candidate_id=target.candidate_id,
                            legal=True,
                        )

        # Missing comparison evidence is repaired by information gathering,
        # never by guessing that the currently selected candidate is optimal.
        query = str(canonical_query or "").strip()
        if available.has_search_bar is True and query:
            return PlanResult(
                action="search[{}]".format(query),
                reason=(
                    "refresh_incomplete_comparison_ledger"
                    if incomplete or target is None
                    else "search_for_preferred_candidate"
                ),
                target_candidate_id=(
                    target.candidate_id if target is not None else None
                ),
                legal=True,
            )

        back = available.matching_clickable("Back to Search")
        if back is not None:
            return PlanResult(
                action="click[{}]".format(back),
                reason=(
                    "return_to_search_for_missing_evidence"
                    if incomplete or target is None
                    else "return_to_search_for_preferred_candidate"
                ),
                target_candidate_id=(
                    target.candidate_id if target is not None else None
                ),
                legal=True,
            )

        return PlanResult(
            action=None,
            reason=(
                "no_legal_information_gathering_action"
                if incomplete or target is None
                else "no_legal_action_toward_preferred_candidate"
            ),
            target_candidate_id=(
                target.candidate_id if target is not None else None
            ),
            legal=False,
        )


def _resolve_current_entry(
    reference: Optional[CandidateReference],
    ledger: Optional[CandidateLedger],
) -> Optional[LedgerEntry]:
    if reference is None:
        return None
    if isinstance(reference, LedgerEntry):
        return reference
    if ledger is not None:
        return ledger.get(reference)
    return None


def _reference_asin(reference: Optional[CandidateReference]) -> Optional[str]:
    if isinstance(reference, Candidate):
        return reference.asin
    if isinstance(reference, LedgerEntry):
        return reference.asin
    return None


@dataclass(frozen=True)
class RuntimeDecision:
    proposed_action: Optional[str]
    executed_action: Optional[str]
    goal_accepted: bool
    choice: ChoiceDecision
    plan: Optional[PlanResult] = None
    reason: str = ""
    repair_certified: Optional[bool] = None

    @property
    def intervened(self) -> bool:
        return self.executed_action != self.proposed_action

    @property
    def blocked(self) -> bool:
        return self.executed_action is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed_action": self.proposed_action,
            "executed_action": self.executed_action,
            "goal_accepted": self.goal_accepted,
            "intervened": self.intervened,
            "blocked": self.blocked,
            "reason": self.reason,
            "repair_certified": self.repair_certified,
            "choice": self.choice.to_dict(),
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


class ChoiceIntegrityDefense:
    """Compose a prior hard-goal decision with the choice-integrity guard.

    Any action synthesized after the proposal must be accepted explicitly by
    ``repair_certifier``. Missing, malformed, or failing certification blocks
    the repair; the original proposal is never used as an implicit fallback.
    """

    def __init__(
        self,
        *,
        guard: Optional[ChoiceIntegrityGuard] = None,
        planner: Optional[DeterministicChoicePlanner] = None,
    ) -> None:
        self.guard = guard or ChoiceIntegrityGuard()
        self.planner = planner or DeterministicChoicePlanner(
            require_evidence=self.guard.require_evidence
        )

    def intercept(
        self,
        action: Optional[str],
        *,
        goal_accepted: bool,
        goal_repair: Optional[str] = None,
        selected_candidate: Optional[CandidateReference],
        ledger: Optional[CandidateLedger],
        preference: Optional[Preference],
        available_actions: Any,
        canonical_query: str,
        hard_constraints: Optional[Mapping[str, Any]] = None,
        selected_options: Optional[Mapping[str, str]] = None,
        current_page: Optional[int] = None,
        repair_certifier: Optional[Callable[[str], bool]] = None,
    ) -> RuntimeDecision:
        # ``hard_constraints`` is accepted as part of the common runtime
        # interceptor interface. The preceding GATE goal layer has already
        # certified them; choice logic must not reinterpret or weaken them.
        del hard_constraints
        available = AvailableActions.parse(available_actions)

        # Hard feasibility has lexical priority. A preference-derived action is
        # never considered when the goal layer rejected and supplied no legal,
        # fully certified hard-goal repair.
        effective_action = action
        repair_certified: Optional[bool] = None
        if not goal_accepted:
            repair_is_legal = (
                goal_repair is not None and available.legal_action(goal_repair)
            )
            if repair_is_legal:
                repair_certified = bool(
                    repair_certifier is not None
                    and _certify_repair(
                        repair_certifier,
                        goal_repair,
                    )
                )
            if (
                not repair_is_legal
                or repair_certified is False
            ):
                choice = ChoiceDecision(
                    status=DecisionStatus.REJECT,
                    action=action,
                    reason="hard_goal_rejected",
                    selected_candidate_id=_reference_id(selected_candidate),
                    ablation=self.guard.ablation,
                )
                return RuntimeDecision(
                    proposed_action=action,
                    executed_action=None,
                    goal_accepted=False,
                    choice=choice,
                    reason=(
                        "hard_goal_rejected_without_certified_repair"
                        if repair_certified is False
                        else "hard_goal_rejected_without_legal_repair"
                    ),
                    repair_certified=repair_certified,
                )
            effective_action = goal_repair

        choice = self.guard.certify(
            effective_action,
            selected_candidate=selected_candidate,
            ledger=ledger,
            preference=preference,
        )
        if choice.accepted:
            return RuntimeDecision(
                proposed_action=action,
                executed_action=effective_action,
                goal_accepted=goal_accepted,
                choice=choice,
                reason=(
                    "goal_repair_accepted"
                    if effective_action != action
                    else "action_accepted"
                ),
                repair_certified=repair_certified,
            )

        plan = self.planner.plan(
            ledger=ledger,
            preference=preference,
            available_actions=available,
            canonical_query=canonical_query,
            current_candidate=selected_candidate,
            selected_options=selected_options,
            current_page=current_page,
        )
        executed = plan.action if plan.legal else None
        if executed is not None:
            repair_certified = bool(
                repair_certifier is not None
                and _certify_repair(
                    repair_certifier,
                    executed,
                )
            )
            if not repair_certified:
                executed = None
        return RuntimeDecision(
            proposed_action=action,
            executed_action=executed,
            goal_accepted=goal_accepted,
            choice=choice,
            plan=plan,
            reason=(
                "choice_rejected_and_uncertified_repair_blocked"
                if repair_certified is False
                else "choice_rejected_and_repaired"
                if executed is not None
                else "choice_rejected_and_blocked"
            ),
            repair_certified=repair_certified,
        )


def _certify_repair(
    certifier: Callable[[str], bool],
    action: str,
) -> bool:
    """Fail closed when an external full-certificate check cannot accept."""

    try:
        return certifier(action) is True
    except Exception:
        return False
