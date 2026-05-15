"""Regex masking for Gate defense."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .goal_contract import GoalContract


_ALWAYS_KEEP = {
    # Prompt/agent protocol words.
    "observation", "available", "actions", "action", "search", "click", "keywords",
    # WebShop/navigation/action words.
    "back", "next", "previous", "prev", "buy", "now", "reviews", "review", "description",
    "features", "feature", "price", "rating", "ratings", "options", "option", "select",
    "size", "color", "colour", "quantity", "cart", "home", "page", "product", "products",
    # Common separators/field names seen in WebShop text.
    "sep", "item", "items", "name", "title", "brand", "stars", "star",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\$?\d+(?:\.\d+)?|[A-Z0-9]{10}")
_PRODUCT_ID_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\$?\d+(?:\.\d+)?$")


@dataclass
class MaskRecord:
    token: str
    start: int
    end: int
    reason: str = "not_related_to_structured_goal"

    def to_dict(self) -> Dict[str, object]:
        return {
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
        }


class RegexGoalMasker:
    """
    Masks word-like tokens not related to the structured goal.
    """

    def __init__(self, mask_token: str = "__", extra_keep_words: Sequence[str] | None = None) -> None:
        self.mask_token = mask_token
        self.extra_keep_words = {w.lower() for w in (extra_keep_words or []) if w}

    def _goal_terms(self, goal: GoalContract) -> Set[str]:
        terms: Set[str] = set()

        def add_value(value):
            if value is None:
                return
            if isinstance(value, dict):
                for k, v in value.items():
                    add_value(k)
                    add_value(v)
                return
            if isinstance(value, (list, tuple, set)):
                for x in value:
                    add_value(x)
                return
            for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9'_-]*|\$?\d+(?:\.\d+)?", str(value).lower()):
                if tok:
                    terms.add(tok.strip("'_- "))

        add_value(getattr(goal, "intent", None))
        add_value(getattr(goal, "positive_constraints", None))
        add_value(getattr(goal, "negative_constraints", None))

        # Compatibility with the previous WebShop-specific StructuredGoal.
        add_value(getattr(goal, "product_type", None))
        add_value(getattr(goal, "attributes", None))
        add_value(getattr(goal, "constraints", None))
        add_value(getattr(goal, "positive_keywords", None))
        add_value(getattr(goal, "negative_keywords", None))
        return {t for t in terms if t}

    @staticmethod
    def _stem_variants(term: str) -> Set[str]:
        variants = {term}
        if term.endswith("s") and not term.endswith(("as", "is", "us", "ss")) and len(term) > 3:
            variants.add(term[:-1])
        if not term.endswith("s") and len(term) > 2:
            variants.add(term + "s")
        if term.endswith("ies") and len(term) > 4:
            variants.add(term[:-3] + "y")
        if term.endswith("y") and len(term) > 3:
            variants.add(term[:-1] + "ies")
        return variants

    def _allowed_terms(self, goal: GoalContract) -> Set[str]:
        allowed = set(_ALWAYS_KEEP) | self.extra_keep_words
        for term in self._goal_terms(goal):
            allowed.update(self._stem_variants(term.lower()))
        return allowed

    def _keep_token(self, token: str, allowed: Set[str]) -> bool:
        raw = token.strip()
        low = raw.lower().strip("'_- ")
        if not low:
            return True
        if _PRODUCT_ID_RE.match(raw):
            return True
        if _NUMBER_RE.match(raw):
            return True
        if low in allowed:
            return True
        # Keep goal-related compounds, e.g. "running-shoes" when "running" or
        # "shoes" is allowed.
        pieces = [p for p in re.split(r"[-_']+", low) if p]
        if pieces and any(p in allowed for p in pieces):
            return True
        return False

    def mask(self, text: str, goal: GoalContract) -> Tuple[str, List[MaskRecord]]:
        if not text:
            return text, []

        allowed = self._allowed_terms(goal)
        records: List[MaskRecord] = []
        chunks: List[str] = []
        last = 0

        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            start, end = match.span()
            chunks.append(text[last:start])
            if self._keep_token(token, allowed):
                chunks.append(token)
            else:
                chunks.append(self.mask_token)
                records.append(MaskRecord(token=token, start=start, end=end))
            last = end

        chunks.append(text[last:])
        return "".join(chunks), records
