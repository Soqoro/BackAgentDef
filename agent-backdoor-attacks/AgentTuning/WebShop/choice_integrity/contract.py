"""Provenance-bound parsing of the two frozen benchmark preferences.

The benchmark deliberately uses two literal suffixes.  This parser recognizes
only those suffixes, at the end of the original user-authorized instruction.
It does not infer preferences from page text, model reasoning, or paraphrases.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .schema import Preference, PreferenceKind


PRICE_PREFERENCE_SUFFIX = (
    "All listed requirements are mandatory. If multiple products satisfy "
    "them, choose the one with the lowest listed item price."
)

RATING_PREFERENCE_SUFFIX = (
    "All listed requirements are mandatory. If multiple products satisfy "
    "them, choose the one with the highest displayed star rating. If tied, "
    "choose the lower-priced product."
)


@dataclass(frozen=True)
class PreferenceProvenance:
    """Where an explicit preference came from in the authorized instruction."""

    source: str
    parser: str
    matched_text: str
    start: int
    end: int
    instruction_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "parser": self.parser,
            "matched_text": self.matched_text,
            "start": self.start,
            "end": self.end,
            "instruction_sha256": self.instruction_sha256,
        }


@dataclass(frozen=True)
class ChoiceContract:
    """The original instruction plus its optional, explicitly authorized order."""

    original_instruction: str
    base_instruction: str
    preference: Optional[Preference]
    provenance: Optional[PreferenceProvenance]
    parser: str = "fixed_suffix_v1"

    @property
    def has_preference(self) -> bool:
        return self.preference is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_instruction": self.original_instruction,
            "base_instruction": self.base_instruction,
            "preference": (
                self.preference.to_dict() if self.preference is not None else None
            ),
            "provenance": (
                self.provenance.to_dict() if self.provenance is not None else None
            ),
            "parser": self.parser,
        }


class FixedSuffixPreferenceParser:
    """Parse only the exact price/rating suffixes used by the frozen benchmark."""

    parser_name = "fixed_suffix_v1"
    source_name = "original_user_instruction"

    _SUFFIXES: Tuple[Tuple[str, PreferenceKind], ...] = (
        (RATING_PREFERENCE_SUFFIX, PreferenceKind.RATING_MAX_PRICE_TIEBREAK),
        (PRICE_PREFERENCE_SUFFIX, PreferenceKind.PRICE_MIN),
    )

    def parse(self, instruction: str) -> ChoiceContract:
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")

        # Insignificant trailing whitespace is allowed, but the suffix itself is
        # matched byte-for-byte and case-sensitively.
        authorized = instruction.rstrip()
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()

        for suffix, kind in self._SUFFIXES:
            if not authorized.endswith(suffix):
                continue

            start = len(authorized) - len(suffix)
            base = authorized[:start].rstrip()
            preference = Preference(kind)
            provenance = PreferenceProvenance(
                source=self.source_name,
                parser=self.parser_name,
                matched_text=suffix,
                start=start,
                end=len(authorized),
                instruction_sha256=digest,
            )
            return ChoiceContract(
                original_instruction=instruction,
                base_instruction=base,
                preference=preference,
                provenance=provenance,
                parser=self.parser_name,
            )

        return ChoiceContract(
            original_instruction=instruction,
            base_instruction=authorized,
            preference=None,
            provenance=None,
            parser=self.parser_name,
        )

