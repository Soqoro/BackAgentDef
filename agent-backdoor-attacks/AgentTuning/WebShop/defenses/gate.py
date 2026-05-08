"""Main Gate defense composition: parse goal -> mask prompt -> report masks."""

from __future__ import annotations

from typing import Optional, Tuple

from .goal_parser import OpenAIGoalParser, RegexGoalParser, StructuredGoal
from .masker import RegexGoalMasker
from .reporter import GateReport


def _preview(text: str, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text)
    if max_chars is None or max_chars < 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[TRUNCATED {len(text) - max_chars} chars]"


class GateDefense:
    """
    Usage:
        gate = GateDefense(use_openai=True)
        gate.start_episode(goal_text)
        masked_prompt, report = gate.apply(prompt_text)
    """

    def __init__(
        self,
        use_openai: bool = True,
        openai_model: str = "gpt-4o-mini",
        mask_token: str = "__",
        report_preview_chars: int = 1200,
    ) -> None:
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.report_preview_chars = report_preview_chars
        self.masker = RegexGoalMasker(mask_token=mask_token)
        self.regex_parser = RegexGoalParser()
        self.openai_parser = OpenAIGoalParser(model=openai_model, fallback_parser=self.regex_parser)
        self.current_goal: Optional[StructuredGoal] = None

    def start_episode(self, instruction: str) -> StructuredGoal:
        parser = self.openai_parser if self.use_openai else self.regex_parser
        self.current_goal = parser.parse(instruction or "")
        return self.current_goal

    def apply(self, text: str, instruction: Optional[str] = None) -> Tuple[str, GateReport]:
        if instruction is not None or self.current_goal is None:
            self.start_episode(instruction or "")
        assert self.current_goal is not None
        masked_text, records = self.masker.mask(text or "", self.current_goal)
        report = GateReport(
            structured_goal=self.current_goal,
            mask_records=records,
            original_text_preview=_preview(text or "", self.report_preview_chars),
            masked_text_preview=_preview(masked_text, self.report_preview_chars),
        )
        return masked_text, report
