"""Reporting utilities for Gate defense."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .goal_parser import StructuredGoal
from .masker import MaskRecord


@dataclass
class GateReport:
    structured_goal: StructuredGoal
    mask_records: List[MaskRecord] = field(default_factory=list)
    original_text_preview: Optional[str] = None
    masked_text_preview: Optional[str] = None

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
        return {
            "structured_goal": self.structured_goal.to_dict(),
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
