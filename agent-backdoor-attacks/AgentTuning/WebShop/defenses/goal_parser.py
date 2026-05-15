"""
Goal parsing for the Gate defense.

The parser converts user instruction into a small structured goal
object that downstream masking can use.  The OpenAI parser is intentionally
optional: if the openai package/API key is unavailable, the code falls back to a
local regex parser so evaluations can still run.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "buy", "by", "for", "from",
    "find", "get", "give", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "please", "purchase", "show", "that", "the", "this", "to", "under",
    "want", "with", "you", "your",
}


@dataclass
class StructuredGoal:
    raw_instruction: str
    product_type: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    positive_keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    parser: str = "unknown"
    parser_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_instruction: str, parser: str) -> "StructuredGoal":
        def list_of_strings(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, Iterable):
                return [str(x).strip() for x in value if str(x).strip()]
            return []

        constraints = data.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}

        return cls(
            raw_instruction=raw_instruction,
            product_type=(str(data.get("product_type")).strip() or None)
            if data.get("product_type") is not None else None,
            attributes=list_of_strings(data.get("attributes")),
            constraints=constraints,
            positive_keywords=list_of_strings(data.get("positive_keywords")),
            negative_keywords=list_of_strings(data.get("negative_keywords")),
            parser=parser,
        )


class RegexGoalParser:
    """Small dependency-free parser used as fallback and for offline runs."""

    parser_name = "regex"

    def parse(self, instruction: str) -> StructuredGoal:
        text = instruction or ""
        lowered = text.lower()

        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'_-]*|\$?\d+(?:\.\d+)?", lowered)
        keywords = []
        for tok in tokens:
            clean = tok.strip("'_- ")
            if not clean or clean in _STOPWORDS:
                continue
            if clean not in keywords:
                keywords.append(clean)

        product_type = None
        product_patterns = [
            r"\b(sneakers?|shoes?|boots?|sandals?|slippers?)\b",
            r"\b(shirts?|pants?|shorts?|jackets?|coats?|dresses?)\b",
            r"\b(bags?|backpacks?|wallets?)\b",
            r"\b(watches?|headphones?|speakers?|chargers?)\b",
        ]
        for pattern in product_patterns:
            m = re.search(pattern, lowered)
            if m:
                product_type = m.group(1)
                break

        constraints: Dict[str, Any] = {}
        price_match = re.search(r"(?:under|below|less than|at most)\s*\$?\s*(\d+(?:\.\d+)?)", lowered)
        if price_match:
            constraints["max_price"] = price_match.group(1)

        rating_match = re.search(r"(?:at least|over|above)\s*(\d+(?:\.\d+)?)\s*stars?", lowered)
        if rating_match:
            constraints["min_rating"] = rating_match.group(1)

        # Treat adjectives and remaining non-stopword terms as useful attributes.
        attributes = [k for k in keywords if k != product_type]

        return StructuredGoal(
            raw_instruction=text,
            product_type=product_type,
            attributes=attributes,
            constraints=constraints,
            positive_keywords=keywords,
            negative_keywords=[],
            parser=self.parser_name,
        )


class OpenAIGoalParser:
    """
    OpenAI-backed structured goal parser.

    Requires the `openai` package and OPENAI_API_KEY. If either is unavailable
    or parsing fails, this parser returns the regex fallback with parser_error set.
    """

    parser_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: float = 30.0,
        fallback_parser: Optional[RegexGoalParser] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.fallback_parser = fallback_parser or RegexGoalParser()

    def parse(self, instruction: str) -> StructuredGoal:
        instruction = instruction or ""
        if not os.environ.get("OPENAI_API_KEY"):
            goal = self.fallback_parser.parse(instruction)
            goal.parser = "regex_fallback"
            goal.parser_error = "OPENAI_API_KEY is not set"
            return goal

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            goal = self.fallback_parser.parse(instruction)
            goal.parser = "regex_fallback"
            goal.parser_error = f"openai package unavailable: {type(exc).__name__}: {exc}"
            return goal

        schema_prompt = """
Convert the WebShop shopping instruction into a compact JSON object.
Return JSON only, with these keys:
- product_type: the main item type, string or null
- attributes: useful descriptive attributes, list of strings
- constraints: constraints such as max_price, min_rating, size, color, material, list/dict values allowed
- positive_keywords: words/phrases that must remain visible for completing the user goal
- negative_keywords: words/phrases that conflict with the user goal or should not guide the agent
""".strip()

        try:
            client = OpenAI(timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": schema_prompt},
                    {"role": "user", "content": instruction},
                ],
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            goal = StructuredGoal.from_dict(data, raw_instruction=instruction, parser=self.parser_name)
        except Exception as exc:
            goal = self.fallback_parser.parse(instruction)
            goal.parser = "regex_fallback"
            goal.parser_error = f"{type(exc).__name__}: {exc}"
            return goal

        # Always add literal instruction tokens as a safety net for masking.
        fallback_goal = self.fallback_parser.parse(instruction)
        merged = []
        for tok in goal.positive_keywords + fallback_goal.positive_keywords:
            if tok and tok.lower() not in [x.lower() for x in merged]:
                merged.append(tok)
        goal.positive_keywords = merged
        return goal


# Module 1 now lives in goal_contract.py. Re-export the new implementation
# under the old parser names so older experiment scripts keep working.
from .goal_contract import (  # noqa: E402
    GoalContract,
    GoalContractExtraction,
    OpenAIGoalContractExtractor,
    RegexGoalContractExtractor,
)

StructuredGoal = GoalContract
RegexGoalParser = RegexGoalContractExtractor
OpenAIGoalParser = OpenAIGoalContractExtractor
