"""
Module 2: Goal-Relevant State Abstraction.

This module maps a raw environment observation o_t to a structured state S_t,
then constructs a neutralized state S_tilde using the goal contract G(q).
For WebShop, the raw observation is prompt-shaped text containing observation
content and available web actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .goal_contract import GoalContract
from .masker import MaskRecord


_ALWAYS_KEEP = {
    # Prompt/agent protocol words.
    "observation", "available", "actions", "action", "search", "click", "keywords",
    # WebShop navigation/action words.
    "back", "next", "previous", "prev", "buy", "now", "reviews", "review",
    "description", "features", "feature", "price", "rating", "ratings",
    "options", "option", "select", "size", "color", "colour", "quantity",
    "cart", "home", "page", "product", "products",
    # Common field names and separators.
    "sep", "item", "items", "name", "title", "brand", "stars", "star",
    "true", "false", "none", "clickables", "has_search_bar",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\$?\d+(?:\.\d+)?|[A-Z0-9]{10}")
_PRODUCT_ID_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\$?\d+(?:\.\d+)?$")
_ACTION_RE = re.compile(r"\b(search|click)\[[^\]]+\]", re.IGNORECASE)
_PRICE_RE = re.compile(r"\$\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:dollars?|usd)\b", re.I)
_RATING_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:stars?|out of 5)\b", re.I)


@dataclass
class StateElement:
    """A typed state element e_i = (tau_i, A_i)."""

    element_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    start: int = -1
    end: int = -1

    @property
    def tau(self) -> str:
        return self.element_type

    @property
    def A(self) -> Dict[str, Any]:
        return self.attributes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau": self.element_type,
            "A": self.attributes,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class StructuredState:
    """Structured state S_t = {e_1, ..., e_n}."""

    elements: List[StateElement] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self, max_elements: Optional[int] = None) -> Dict[str, Any]:
        elements = self.elements
        truncated = 0
        if max_elements is not None and len(elements) > max_elements:
            truncated = len(elements) - max_elements
            elements = elements[:max_elements]

        return {
            "elements": [element.to_dict() for element in elements],
            "element_count": len(self.elements),
            "truncated_elements": truncated,
        }


@dataclass
class StateAbstractionResult:
    """Output of Module 2: S_t, S_tilde, and the neutralized prompt text."""

    structured_state: StructuredState
    neutralized_state: StructuredState
    neutralized_text: str
    mask_records: List[MaskRecord] = field(default_factory=list)
    relevance_terms: List[str] = field(default_factory=list)
    mask_token: str = "__"

    @property
    def mask_count(self) -> int:
        return len(self.mask_records)

    def to_dict(self, max_elements: int = 40, max_records: int = 50) -> Dict[str, Any]:
        return {
            "module": GoalRelevantStateAbstraction.module_name,
            "structured_state": self.structured_state.to_dict(max_elements=max_elements),
            "neutralized_state": self.neutralized_state.to_dict(max_elements=max_elements),
            "mask_count": self.mask_count,
            "mask_records_preview": [
                record.to_dict() for record in self.mask_records[:max_records]
            ],
            "relevance_terms": list(self.relevance_terms),
            "mask_token": self.mask_token,
        }


class GoalRelevantStateAbstraction:
    """WebShop implementation of f_state(o_t) and f_abs(S_t, G(q))."""

    module_name = "goal_relevant_state_abstraction"

    def __init__(
        self,
        mask_token: str = "__",
        extra_keep_words: Optional[Sequence[str]] = None,
    ) -> None:
        self.mask_token = mask_token
        self.extra_keep_words = {
            word.lower()
            for word in (extra_keep_words or [])
            if word
        }

    def abstract(self, observation: str, goal_contract: GoalContract) -> StateAbstractionResult:
        state = self.f_state(observation or "")
        return self.f_abs(state, goal_contract)

    def f_state(self, observation: str) -> StructuredState:
        """Convert raw WebShop prompt text into typed state elements."""

        raw_text = observation or ""
        elements: List[StateElement] = []
        current_section = "raw"
        offset = 0

        for line in raw_text.splitlines(keepends=True):
            start = offset
            offset += len(line)
            line_text = line.rstrip("\r\n")
            stripped = line_text.strip()

            if not stripped:
                continue

            normalized_header = stripped.rstrip(":").lower()
            if normalized_header == "observation":
                current_section = "observation"
                elements.append(
                    StateElement(
                        element_type="section_header",
                        attributes={"label": "Observation"},
                        start=start,
                        end=start + len(line_text),
                    )
                )
                continue

            if normalized_header == "available actions":
                current_section = "available_actions"
                elements.append(
                    StateElement(
                        element_type="section_header",
                        attributes={"label": "Available Actions"},
                        start=start,
                        end=start + len(line_text),
                    )
                )
                continue

            attributes = self._line_attributes(stripped, current_section)
            elements.append(
                StateElement(
                    element_type=self._infer_element_type(stripped, current_section),
                    attributes=attributes,
                    start=start,
                    end=start + len(line_text),
                )
            )

        if raw_text and not elements:
            elements.append(
                StateElement(
                    element_type="web_text",
                    attributes={"section": "raw", "text": raw_text},
                    start=0,
                    end=len(raw_text),
                )
            )

        return StructuredState(elements=elements, raw_text=raw_text)

    def f_abs(
        self,
        state: StructuredState,
        goal_contract: GoalContract,
    ) -> StateAbstractionResult:
        """
        Construct S_tilde by masking attributes not grounded in I or C+.
        """

        allowed_terms = self._allowed_terms(goal_contract)
        neutralized_text, mask_records = self._neutralize_text(
            state.raw_text,
            allowed_terms,
            base_offset=0,
            collect_records=True,
        )
        neutralized_elements = [
            StateElement(
                element_type=element.element_type,
                attributes=self._neutralize_value(element.attributes, allowed_terms),
                start=element.start,
                end=element.end,
            )
            for element in state.elements
        ]

        return StateAbstractionResult(
            structured_state=state,
            neutralized_state=StructuredState(
                elements=neutralized_elements,
                raw_text=neutralized_text,
            ),
            neutralized_text=neutralized_text,
            mask_records=mask_records,
            relevance_terms=sorted(allowed_terms),
            mask_token=self.mask_token,
        )

    def _line_attributes(self, text: str, section: str) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {
            "section": section,
            "text": text,
        }

        parts = [part.strip() for part in text.split("[SEP]") if part.strip()]
        if len(parts) > 1:
            attributes["parts"] = parts

        product_ids = re.findall(r"\b[A-Z0-9]{10}\b", text, flags=re.I)
        if product_ids:
            attributes["product_ids"] = product_ids

        actions = [match.group(0) for match in _ACTION_RE.finditer(text)]
        if actions:
            attributes["actions"] = actions

        prices = [match.group(0) for match in _PRICE_RE.finditer(text)]
        if prices:
            attributes["prices"] = prices

        ratings = [match.group(0) for match in _RATING_RE.finditer(text)]
        if ratings:
            attributes["ratings"] = ratings

        return attributes

    def _infer_element_type(self, text: str, section: str) -> str:
        if section == "available_actions" or _ACTION_RE.search(text):
            return "web_action"
        if "[SEP]" in text or re.search(r"\b[A-Z0-9]{10}\b", text, flags=re.I):
            return "web_element"
        if _PRICE_RE.search(text) or _RATING_RE.search(text):
            return "product_attribute"
        return "web_text"

    def _neutralize_value(self, value: Any, allowed_terms: Set[str]) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            neutralized, _ = self._neutralize_text(
                value,
                allowed_terms,
                base_offset=-1,
                collect_records=False,
            )
            return neutralized
        if isinstance(value, list):
            return [self._neutralize_value(item, allowed_terms) for item in value]
        if isinstance(value, tuple):
            return tuple(self._neutralize_value(item, allowed_terms) for item in value)
        if isinstance(value, set):
            return {
                self._neutralize_value(item, allowed_terms)
                for item in value
            }
        if isinstance(value, dict):
            return {
                key: self._neutralize_value(item, allowed_terms)
                for key, item in value.items()
            }
        return value

    def _neutralize_text(
        self,
        text: str,
        allowed_terms: Set[str],
        base_offset: int,
        collect_records: bool,
    ) -> Tuple[str, List[MaskRecord]]:
        if not text:
            return text, []

        records: List[MaskRecord] = []
        chunks: List[str] = []
        last = 0

        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            start, end = match.span()
            chunks.append(text[last:start])

            if self._keep_token(token, allowed_terms):
                chunks.append(token)
            else:
                chunks.append(self.mask_token)
                if collect_records:
                    records.append(
                        MaskRecord(
                            token=token,
                            start=base_offset + start,
                            end=base_offset + end,
                            reason="not_relevant_to_goal_contract",
                        )
                    )

            last = end

        chunks.append(text[last:])
        return "".join(chunks), records

    def _allowed_terms(self, goal_contract: GoalContract) -> Set[str]:
        terms = set(_ALWAYS_KEEP) | self.extra_keep_words

        for term in self._goal_terms(goal_contract):
            terms.update(self._stem_variants(term.lower()))

        return {term for term in terms if term}

    def _goal_terms(self, goal_contract: GoalContract) -> Set[str]:
        terms: Set[str] = set()
        self._add_terms(terms, goal_contract.intent)
        self._add_terms(terms, goal_contract.positive_constraints)
        return terms

    def _add_terms(self, terms: Set[str], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._add_terms(terms, key)
                self._add_terms(terms, item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._add_terms(terms, item)
            return

        for token in _TOKEN_RE.findall(str(value).lower()):
            token = token.strip("'_- ")
            if token:
                terms.add(token)

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

    def _keep_token(self, token: str, allowed_terms: Set[str]) -> bool:
        raw = token.strip()
        lowered = raw.lower().strip("'_- ")

        if not lowered:
            return True
        if _PRODUCT_ID_RE.match(raw):
            return True
        if _NUMBER_RE.match(raw):
            return True
        if lowered in allowed_terms:
            return True

        pieces = [piece for piece in re.split(r"[-_']+", lowered) if piece]
        if pieces and any(piece in allowed_terms for piece in pieces):
            return True

        return False
