"""Main Gate defense composition for the staged goal-grounded defense."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

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


@dataclass(frozen=True)
class GateModuleConfig:
    """Enable/disable switches for staged GATE ablation runs."""

    goal_contract_extraction: bool = True
    state_abstraction: bool = True
    action_certification: bool = True
    action_projection: bool = True
    output_masking: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


GATE_ABLATIONS: Dict[str, GateModuleConfig] = {
    "full": GateModuleConfig(),
    "no_m1_goal_contract": GateModuleConfig(goal_contract_extraction=False),
    "no_m2_state_abstraction": GateModuleConfig(state_abstraction=False),
    # Module 4 is only invoked after Module 3 rejects an action, so removing
    # Module 3 also makes projection unreachable for that run.
    "no_m3_action_certification": GateModuleConfig(
        action_certification=False,
        action_projection=False,
    ),
    # Output masking also rewrites search[...] actions after generation, so
    # disable it here to keep this ablation free of action rewrites.
    "no_m4_action_projection": GateModuleConfig(
        action_projection=False,
        output_masking=False,
    ),
    "no_m2_no_m4": GateModuleConfig(
        state_abstraction=False,
        action_projection=False,
        output_masking=False,
    ),
    "no_output_masking": GateModuleConfig(output_masking=False),
}

GATE_ABLATION_CHOICES = tuple(GATE_ABLATIONS.keys())
GATE_RUNTIME_MODE_CHOICES = ("full", "mask_only", "enforce_only")


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
        ablation: str = "full",
        runtime_mode: str = "full",
        require_goal_parser_success: bool = False,
        goal_contract_cache_path: Optional[str] = None,
    ) -> None:
        if ablation not in GATE_ABLATIONS:
            choices = ", ".join(GATE_ABLATION_CHOICES)
            raise ValueError(f"Unknown Gate ablation '{ablation}'. Choices: {choices}")
        if runtime_mode not in GATE_RUNTIME_MODE_CHOICES:
            choices = ", ".join(GATE_RUNTIME_MODE_CHOICES)
            raise ValueError(f"Unknown Gate runtime mode '{runtime_mode}'. Choices: {choices}")
        if runtime_mode != "full" and ablation != "full":
            raise ValueError(
                "mask_only/enforce_only runtime modes require the full Gate module set"
            )

        self.use_openai = use_openai
        self.openai_model = openai_model
        self.report_preview_chars = report_preview_chars
        self.ablation = ablation
        self.runtime_mode = runtime_mode
        self.modules = GATE_ABLATIONS[ablation]
        self.masker = RegexGoalMasker(mask_token=mask_token)
        self.state_abstraction = GoalRelevantStateAbstraction(mask_token=mask_token)
        self.action_certification = GoalGroundedActionCertification()
        self.action_projection = GoalConstrainedActionProjection(
            certifier=self.action_certification,
        )
        self.goal_contract_extraction = GoalContractExtraction(
            use_openai=use_openai,
            openai_model=openai_model,
            require_success=require_goal_parser_success,
            cache_path=goal_contract_cache_path,
        )
        self.current_goal_contract: Optional[GoalContract] = None
        self.last_state_abstraction_result: Optional[StateAbstractionResult] = None

        # Backward-compatible attribute names for older experiment code.
        self.regex_parser = self.goal_contract_extraction.regex_extractor
        self.openai_parser = self.goal_contract_extraction.openai_extractor
        self.current_goal: Optional[GoalContract] = None

    def start_episode(
        self,
        instruction: str,
        *,
        goal_only_instruction: Optional[str] = None,
        comparative_preference: Optional[Dict[str, object]] = None,
        preference_provenance: Optional[Dict[str, object]] = None,
    ) -> GoalContract:
        if self.runtime_mode == "full" and not self.modules.goal_contract_extraction:
            self.current_goal_contract = self._disabled_goal_contract(instruction or "")
            self.current_goal = self.current_goal_contract
            return self.current_goal_contract

        self.goal_contract_extraction.use_openai = self.use_openai
        extraction_input = (
            goal_only_instruction
            if goal_only_instruction is not None
            else instruction
        )
        self.current_goal_contract = self.goal_contract_extraction.extract(
            extraction_input or ""
        )
        # Preserve the complete authorized instruction while keeping the
        # deterministic comparative suffix out of the hard-constraint parser.
        self.current_goal_contract.raw_query = instruction or ""
        self.current_goal_contract.comparative_preference = (
            dict(comparative_preference)
            if comparative_preference is not None
            else None
        )
        self.current_goal_contract.preference_provenance = (
            dict(preference_provenance)
            if preference_provenance is not None
            else None
        )
        self.current_goal = self.current_goal_contract
        return self.current_goal_contract

    def apply(self, text: str, instruction: Optional[str] = None) -> Tuple[str, GateReport]:
        if instruction is not None or self.current_goal_contract is None:
            self.start_episode(instruction or "")
        assert self.current_goal_contract is not None

        if self.runtime_mode == "enforce_only":
            state = self.state_abstraction.f_state(text or "")
            state_result = StateAbstractionResult(
                structured_state=state,
                neutralized_state=state,
                neutralized_text=text or "",
                mask_records=[],
                relevance_terms=[],
                mask_token=self.state_abstraction.mask_token,
            )
        elif self.modules.state_abstraction:
            state_result = self.state_abstraction.abstract(text or "", self.current_goal_contract)
        else:
            state = self.state_abstraction.f_state(text or "")
            state_result = StateAbstractionResult(
                structured_state=state,
                neutralized_state=state,
                neutralized_text=text or "",
                mask_records=[],
                relevance_terms=[],
                mask_token=self.state_abstraction.mask_token,
            )

        self.last_state_abstraction_result = state_result
        report = GateReport(
            goal_contract=self.current_goal_contract,
            mask_records=state_result.mask_records,
            state_abstraction_result=state_result,
            original_text_preview=_preview(text or "", self.report_preview_chars),
            masked_text_preview=_preview(state_result.neutralized_text, self.report_preview_chars),
        )
        return state_result.neutralized_text, report

    def module_config_dict(self) -> Dict[str, bool]:
        if self.runtime_mode == "full":
            return self.modules.to_dict()
        if self.runtime_mode == "mask_only":
            return GateModuleConfig(
                goal_contract_extraction=True,
                state_abstraction=True,
                action_certification=False,
                action_projection=False,
                output_masking=False,
            ).to_dict()
        return GateModuleConfig(
            goal_contract_extraction=True,
            state_abstraction=True,
            action_certification=True,
            action_projection=True,
            output_masking=False,
        ).to_dict()

    def should_certify_action(self) -> bool:
        if self.runtime_mode == "mask_only":
            return False
        if self.runtime_mode == "enforce_only":
            return True
        return self.modules.action_certification

    def should_project_action(self) -> bool:
        if self.runtime_mode == "mask_only":
            return False
        if self.runtime_mode == "enforce_only":
            return True
        return self.modules.action_projection

    def should_mask_output_action(self) -> bool:
        if self.runtime_mode in {"mask_only", "enforce_only"}:
            return False
        return self.modules.output_masking

    def parser_stats_dict(self) -> Dict[str, object]:
        return self.goal_contract_extraction.stats_dict()

    def _disabled_goal_contract(self, instruction: str) -> GoalContract:
        return GoalContract(
            raw_query=instruction or "",
            intent="",
            positive_constraints=[],
            negative_constraints=[],
            extractor=f"gate_ablation:{self.ablation}:disabled",
            extraction_error="Gate Module 1 goal contract extraction disabled by ablation.",
        )

    def certify_action(
        self,
        action: str,
        state: Optional[StructuredState] = None,
        goal_contract: Optional[GoalContract] = None,
        neutralized_state: Optional[StructuredState] = None,
    ) -> ActionCertificationResult:
        """
        Module 3: certify a proposed action against the latest S_t and G(q).

        If state is omitted, this uses the structured state from the most recent
        prompt-side `apply()` call.
        """

        if not self.should_certify_action():
            raise RuntimeError("Gate action certification is disabled by this ablation.")

        goal = goal_contract or self.current_goal_contract
        if goal is None:
            self.start_episode("")
            goal = self.current_goal_contract
        assert goal is not None

        if state is None:
            if self.last_state_abstraction_result is not None:
                state = self.last_state_abstraction_result.structured_state
                if neutralized_state is None:
                    neutralized_state = (
                        self.last_state_abstraction_result.neutralized_state
                    )
            else:
                state = self.state_abstraction.f_state("")
        if neutralized_state is None:
            neutralized_state = state

        return self.action_certification.certify(
            action or "",
            state,
            goal,
            neutralized_state=neutralized_state,
        )

    def project_action(
        self,
        action: str,
        legal_actions,
        certification_result: Optional[ActionCertificationResult] = None,
        state: Optional[StructuredState] = None,
        goal_contract: Optional[GoalContract] = None,
        neutralized_state: Optional[StructuredState] = None,
    ) -> ActionProjectionResult:
        """
        Module 4: project a rejected action into a legal goal-constrained action.

        If state is omitted, this uses the structured state from the most recent
        prompt-side `apply()` call.
        """

        if not self.should_project_action():
            raise RuntimeError("Gate action projection is disabled by this ablation.")

        goal = goal_contract or self.current_goal_contract
        if goal is None:
            self.start_episode("")
            goal = self.current_goal_contract
        assert goal is not None

        if state is None:
            if self.last_state_abstraction_result is not None:
                state = self.last_state_abstraction_result.structured_state
                if neutralized_state is None:
                    neutralized_state = (
                        self.last_state_abstraction_result.neutralized_state
                    )
            else:
                state = self.state_abstraction.f_state("")
        if neutralized_state is None:
            neutralized_state = state

        return self.action_projection.project(
            action_text=action or "",
            legal_actions=legal_actions,
            structured_state=state,
            neutralized_state=neutralized_state,
            goal_contract=goal,
            certification_result=certification_result,
        )
