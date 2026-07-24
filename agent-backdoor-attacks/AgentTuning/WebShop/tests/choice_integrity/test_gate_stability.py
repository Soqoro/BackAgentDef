import unittest
from types import SimpleNamespace

from choice_integrity.contract import (
    PRICE_PREFERENCE_SUFFIX,
    RATING_PREFERENCE_SUFFIX,
)
from choice_integrity.experiment import _gate_action
from defenses.action_certification import (
    ActionCertificationResult,
    CertificationCheck,
    ProposedAction,
)
from defenses.gate import GateDefense
from defenses.goal_contract import GoalContract


PROMPT = """Observation:
Adidas black sneaker

Available Actions:
{'has_search_bar': True, 'clickables': ['Adidas']}
"""


def certificate(
    action,
    *,
    support=True,
    progress=True,
    safety=True,
    stability=True,
):
    operation, _, target = action.partition("[")
    return ActionCertificationResult(
        proposed_action=ProposedAction(
            raw_action=action,
            operation=operation,
            target=target.rstrip("]"),
        ),
        support=CertificationCheck("Supp", support),
        progress=CertificationCheck("Prog", progress),
        safety=CertificationCheck("Safe", safety),
        stability=CertificationCheck("Stable", stability),
    )


class GateStabilityTests(unittest.TestCase):
    def test_goal_contract_keeps_structured_hard_constraints(self):
        contract = GoalContract.from_dict(
            {
                "I": "buy a sneaker",
                "C_plus": ["black", "size medium", "under $80"],
                "C_minus": [],
                "product_type": "sneaker",
                "attributes": ["black"],
                "options": {"size": "medium"},
                "max_price": 80,
                "min_rating": None,
            },
            raw_query="Find a black sneaker with size: medium under $80.",
            extractor="test",
        )

        self.assertEqual(contract.product_type, "sneaker")
        self.assertEqual(contract.attributes, ["black"])
        self.assertEqual(contract.options, {"size": "medium"})
        self.assertEqual(contract.max_price, 80.0)
        self.assertEqual(contract.G["options"], {"size": "medium"})

    def test_trigger_dependent_action_fails_counterfactual_stability(self):
        gate = GateDefense(use_openai=False)
        gate.start_episode("Find a black sneaker.")
        gate.apply(PROMPT)

        result = gate.certify_action("click[Adidas]")

        self.assertTrue(result.support.passed)
        self.assertTrue(result.progress.passed)
        self.assertFalse(result.stability.passed)
        self.assertFalse(result.accepted)
        self.assertIn("Stable", result.to_dict())

    def test_projection_uses_same_neutralized_state(self):
        gate = GateDefense(use_openai=False)
        gate.start_episode("Find a black sneaker.")
        gate.apply(PROMPT)
        original = gate.certify_action("click[Adidas]")

        projection = gate.project_action(
            action="click[Adidas]",
            legal_actions={
                "has_search_bar": True,
                "clickables": ["Adidas"],
            },
            certification_result=original,
        )

        self.assertNotEqual(projection.projected_action, "click[Adidas]")
        self.assertTrue(projection.projected_certification.stability.passed)

    def test_preference_is_bound_separately_from_hard_constraints(self):
        gate = GateDefense(use_openai=False)
        contract = gate.start_episode(
            "Find a black sneaker. Choose the cheapest.",
            goal_only_instruction="Find a black sneaker.",
            comparative_preference={"kind": "price_min"},
            preference_provenance={"source": "original_user_instruction"},
        )

        self.assertEqual(contract.comparative_preference, {"kind": "price_min"})
        self.assertEqual(
            contract.G["P_provenance"]["source"],
            "original_user_instruction",
        )
        self.assertNotIn("cheapest", repr(contract.positive_constraints).lower())

    def test_price_preference_vocabulary_survives_abstraction(self):
        gate = GateDefense(use_openai=False)
        gate.start_episode(
            f"Find a black sneaker. {PRICE_PREFERENCE_SUFFIX}",
            goal_only_instruction="Find a black sneaker.",
            comparative_preference={"kind": "price_min"},
        )

        neutralized, _ = gate.apply(
            f"Observation:\n{PRICE_PREFERENCE_SUFFIX} Adidas\n"
        )

        self.assertIn(PRICE_PREFERENCE_SUFFIX, neutralized)
        self.assertNotIn("Adidas", neutralized)

    def test_rating_tiebreak_vocabulary_survives_abstraction(self):
        gate = GateDefense(use_openai=False)
        gate.start_episode(
            f"Find a black sneaker. {RATING_PREFERENCE_SUFFIX}",
            goal_only_instruction="Find a black sneaker.",
            comparative_preference={"kind": "rating_max_price_tiebreak"},
        )

        neutralized, _ = gate.apply(
            f"Observation:\n{RATING_PREFERENCE_SUFFIX} Adidas\n"
        )

        self.assertIn(RATING_PREFERENCE_SUFFIX, neutralized)
        self.assertNotIn("Adidas", neutralized)

    def test_unknown_preference_kind_does_not_whitelist_vocabulary(self):
        gate = GateDefense(use_openai=False)
        gate.start_episode(
            "Find a black sneaker.",
            goal_only_instruction="Find a black sneaker.",
            comparative_preference={"kind": "adidas"},
        )

        neutralized, _ = gate.apply(
            f"Observation:\n{PRICE_PREFERENCE_SUFFIX} Adidas\n"
        )

        self.assertNotIn("lowest", neutralized.lower())
        self.assertNotIn("Adidas", neutralized)

    def test_output_masking_restores_page_state_before_recertification(self):
        gate = GateDefense(use_openai=False)
        gate.start_episode("Find a black sneaker.")
        gate.apply(PROMPT)
        page_state = gate.last_state_abstraction_result

        class Conversation:
            def __init__(self):
                self.message = None

            def update_last_message(self, value):
                self.message = value

        chat = type("Chat", (), {"conv": Conversation()})()

        def mutating_output_mask(active_gate, _action):
            active_gate.apply("adidas-only action text")
            return "search[sneaker black]", None

        _gate_action(
            gate=gate,
            chat=chat,
            action="search[black sneaker]",
            response="Action: search[black sneaker]",
            available_actions={
                "has_search_bar": True,
                "clickables": ["Adidas"],
            },
            replace_action=lambda response, original_action, masked_action: (
                response.replace(original_action, masked_action, 1)
            ),
            output_mask=mutating_output_mask,
        )

        self.assertIs(gate.last_state_abstraction_result, page_state)
        self.assertEqual(
            gate.last_state_abstraction_result.structured_state.raw_text,
            PROMPT,
        )

    def test_failed_output_mask_keeps_only_prior_fully_certified_projection(self):
        original_action = "click[Adidas]"
        projected_action = "search[black sneaker]"
        masked_action = "search[unrelated]"
        original_certificate = certificate(
            original_action,
            stability=False,
        )
        projected_certificate = certificate(projected_action)
        # This deliberately passes the three checks that the old
        # ``safe_repair_ok`` inspected while failing progress.
        masked_certificate = certificate(masked_action, progress=False)
        projection = SimpleNamespace(
            changed=True,
            projected_action=projected_action,
            projected_certification=projected_certificate,
            to_dict=lambda: {"accepted": True},
        )

        class FakeGate:
            last_state_abstraction_result = object()

            @staticmethod
            def should_certify_action():
                return True

            @staticmethod
            def should_project_action():
                return True

            @staticmethod
            def should_mask_output_action():
                return True

            @staticmethod
            def certify_action(action):
                return {
                    original_action: original_certificate,
                    masked_action: masked_certificate,
                }[action]

            @staticmethod
            def project_action(**_kwargs):
                return projection

        class Conversation:
            def __init__(self):
                self.message = None

            def update_last_message(self, value):
                self.message = value

        chat = type("Chat", (), {"conv": Conversation()})()
        executed, response, details, accepted = _gate_action(
            gate=FakeGate(),
            chat=chat,
            action=original_action,
            response=f"Action: {original_action}",
            available_actions={
                "has_search_bar": True,
                "clickables": ["Adidas"],
            },
            replace_action=lambda response, original_action, masked_action: (
                response.replace(original_action, masked_action, 1)
            ),
            output_mask=lambda _gate, _action: (masked_action, None),
        )

        self.assertEqual(executed, projected_action)
        self.assertEqual(response, f"Action: {projected_action}")
        self.assertTrue(accepted)
        self.assertTrue(details["safe_repair_accepted"])
        self.assertFalse(details["output_mask"]["accepted"])
        self.assertTrue(details["output_mask"]["blocked"])
        self.assertEqual(
            details["output_mask"]["fallback_action"],
            projected_action,
        )
        self.assertFalse(
            details["output_mask"]["recertification"]["Prog"]["passed"]
        )

    def test_output_mask_blocks_when_no_certified_fallback_exists(self):
        class NoCertificationGate:
            last_state_abstraction_result = object()

            @staticmethod
            def should_certify_action():
                return False

            @staticmethod
            def should_project_action():
                return False

            @staticmethod
            def should_mask_output_action():
                return True

        class Conversation:
            def update_last_message(self, _value):
                raise AssertionError("an uncertified rewrite changed the transcript")

        original_action = "search[black sneaker]"
        executed, response, details, accepted = _gate_action(
            gate=NoCertificationGate(),
            chat=type("Chat", (), {"conv": Conversation()})(),
            action=original_action,
            response=f"Action: {original_action}",
            available_actions={"has_search_bar": True, "clickables": []},
            replace_action=lambda response, original_action, masked_action: (
                response.replace(original_action, masked_action, 1)
            ),
            output_mask=lambda _gate, _action: ("search[unrelated]", None),
        )

        self.assertIsNone(executed)
        self.assertEqual(response, f"Action: {original_action}")
        self.assertFalse(accepted)
        self.assertTrue(details["output_mask"]["blocked"])
        self.assertNotIn("fallback_action", details["output_mask"])


if __name__ == "__main__":
    unittest.main()
