"""Reporting utilities for Gate defense."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .goal_contract import GoalContract
from .masker import MaskRecord
from .state_abstraction import StateAbstractionResult


@dataclass
class GateReport:
    goal_contract: GoalContract
    mask_records: List[MaskRecord] = field(default_factory=list)
    state_abstraction_result: Optional[StateAbstractionResult] = None
    original_text_preview: Optional[str] = None
    masked_text_preview: Optional[str] = None

    @property
    def structured_goal(self) -> GoalContract:
        """Backward-compatible name used by earlier debug logs."""

        return self.goal_contract

    @property
    def mask_count(self) -> int:
        return len(self.mask_records)

    def masked_terms(self, max_terms: Optional[int] = None) -> List[str]:
        seen = []
        for rec in self.mask_records:
            if rec.token not in seen:
                seen.append(rec.token)
            if max_terms is not None and len(seen) >= max_terms:
                break
        return seen

    def to_dict(self, max_terms: int = 50) -> Dict[str, object]:
        contract = self.goal_contract.to_dict()
        return {
            "goal_contract": contract,
            "structured_goal": contract,
            "state_abstraction": (
                self.state_abstraction_result.to_dict(max_records=max_terms)
                if self.state_abstraction_result is not None
                else None
            ),
            "mask_count": self.mask_count,
            "masked_terms_preview": self.masked_terms(max_terms=max_terms),
            "mask_records_preview": [rec.to_dict() for rec in self.mask_records[:max_terms]],
            "original_text_preview": self.original_text_preview,
            "masked_text_preview": self.masked_text_preview,
        }

    def human_summary(self, max_terms: int = 30) -> str:
        terms = self.masked_terms(max_terms=max_terms)
        if not terms:
            return "Gate did not mask any terms."
        suffix = "" if len(terms) == self.mask_count else f" ... (+{self.mask_count - len(terms)} more masks)"
        return f"Gate masked {self.mask_count} tokens: {', '.join(terms)}{suffix}"
