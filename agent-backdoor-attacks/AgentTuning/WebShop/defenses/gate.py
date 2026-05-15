"""Main Gate defense composition for the staged goal-grounded defense."""

from __future__ import annotations

from typing import Optional, Tuple

from .action_certification import ActionCertificationResult, GoalGroundedActionCertification
from .action_projection import ActionProjectionResult, GoalConstrainedActionProjection
from .goal_contract import GoalContract, GoalContractExtraction
from .masker import RegexGoalMasker
from .reporter import GateReport
from .state_abstraction import GoalRelevantStateAbstraction, StateAbstractionResult, StructuredState


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

    Current staged pipeline:
        1. Goal Contract Extraction
        2. Goal-Relevant State Abstraction
        3. Goal-Grounded Action Certification
        4. Goal-Constrained Action Projection
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
        self.state_abstraction = GoalRelevantStateAbstraction(mask_token=mask_token)
        self.action_certification = GoalGroundedActionCertification()
        self.action_projection = GoalConstrainedActionProjection(
            certifier=self.action_certification,
        )
        self.goal_contract_extraction = GoalContractExtraction(
            use_openai=use_openai,
            openai_model=openai_model,
        )
        self.current_goal_contract: Optional[GoalContract] = None
        self.last_state_abstraction_result: Optional[StateAbstractionResult] = None

        # Backward-compatible attribute names for older experiment code.
        self.regex_parser = self.goal_contract_extraction.regex_extractor
        self.openai_parser = self.goal_contract_extraction.openai_extractor
        self.current_goal: Optional[GoalContract] = None

    def start_episode(self, instruction: str) -> GoalContract:
        self.goal_contract_extraction.use_openai = self.use_openai
        self.current_goal_contract = self.goal_contract_extraction.extract(instruction or "")
        self.current_goal = self.current_goal_contract
        return self.current_goal_contract

    def apply(self, text: str, instruction: Optional[str] = None) -> Tuple[str, GateReport]:
        if instruction is not None or self.current_goal_contract is None:
            self.start_episode(instruction or "")
        assert self.current_goal_contract is not None
        state_result = self.state_abstraction.abstract(text or "", self.current_goal_contract)
        self.last_state_abstraction_result = state_result
        report = GateReport(
            goal_contract=self.current_goal_contract,
            mask_records=state_result.mask_records,
            state_abstraction_result=state_result,
            original_text_preview=_preview(text or "", self.report_preview_chars),
            masked_text_preview=_preview(state_result.neutralized_text, self.report_preview_chars),
        )
        return state_result.neutralized_text, report

    def certify_action(
        self,
        action: str,
        state: Optional[StructuredState] = None,
        goal_contract: Optional[GoalContract] = None,
    ) -> ActionCertificationResult:
        """
        Module 3: certify a proposed action against the latest S_t and G(q).

        If state is omitted, this uses the structured state from the most recent
        prompt-side `apply()` call.
        """

        goal = goal_contract or self.current_goal_contract
        if goal is None:
            self.start_episode("")
            goal = self.current_goal_contract
        assert goal is not None

        if state is None:
            if self.last_state_abstraction_result is not None:
                state = self.last_state_abstraction_result.structured_state
            else:
                state = self.state_abstraction.f_state("")

        return self.action_certification.certify(action or "", state, goal)

    def project_action(
        self,
        action: str,
        legal_actions,
        certification_result: Optional[ActionCertificationResult] = None,
        state: Optional[StructuredState] = None,
        goal_contract: Optional[GoalContract] = None,
    ) -> ActionProjectionResult:
        """
        Module 4: project a rejected action into a legal goal-constrained action.

        If state is omitted, this uses the structured state from the most recent
        prompt-side `apply()` call.
        """

        goal = goal_contract or self.current_goal_contract
        if goal is None:
            self.start_episode("")
            goal = self.current_goal_contract
        assert goal is not None

        if state is None:
            if self.last_state_abstraction_result is not None:
                state = self.last_state_abstraction_result.structured_state
            else:
                state = self.state_abstraction.f_state("")

        return self.action_projection.project(
            action_text=action or "",
            legal_actions=legal_actions,
            structured_state=state,
            goal_contract=goal,
            certification_result=certification_result,
        )
