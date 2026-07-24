import unittest
from unittest import mock

from choice_integrity.defense import (
    AvailableActions,
    ChoiceIntegrityDefense,
    ChoiceIntegrityGuard,
    DecisionStatus,
    DeterministicChoicePlanner,
    is_commitment_action,
)
from choice_integrity.ledger import CandidateLedger
from choice_integrity.schema import Candidate, Preference


def candidate(
    asin,
    *,
    options=None,
    feasible=True,
    price=None,
    rating=None,
    page=None,
):
    return Candidate(
        asin=asin,
        options=options or {},
        feasible=feasible,
        price=price,
        rating=rating,
        page=page,
        evidence={"price": {"value": price}, "rating": {"value": rating}},
    )


def price_fixture():
    winner = candidate(
        "WINNER",
        options={"color": "blue", "size": "M"},
        price=10,
        rating=4.4,
        page=1,
    )
    dominated = candidate("DOMINATED", price=20, rating=4.9, page=2)
    infeasible = candidate("CHEAP-BUT-WRONG", feasible=False, price=1, page=1)
    ledger = CandidateLedger.from_candidates(
        (dominated, infeasible, winner),
        source="fixed_shortlist",
    )
    return winner, dominated, infeasible, ledger


class ChoiceIntegrityGuardTests(unittest.TestCase):
    def test_only_buy_now_is_a_commitment(self):
        self.assertTrue(is_commitment_action(" click[Buy Now] "))
        self.assertTrue(is_commitment_action("click[buy   now]"))
        for action in ("click[WINNER]", "search[buy now]", "click[Buy Later]", None):
            with self.subTest(action=action):
                self.assertFalse(is_commitment_action(action))

    def test_noncommitment_and_absent_preference_are_noops(self):
        guard = ChoiceIntegrityGuard()
        noncommit = guard.certify(
            "click[WINNER]",
            selected_candidate=None,
            ledger=None,
            preference=Preference.price_min(),
        )
        no_preference = guard.certify(
            "click[Buy Now]",
            selected_candidate=None,
            ledger=None,
            preference=None,
        )

        self.assertEqual(noncommit.status, DecisionStatus.ACCEPT)
        self.assertEqual(noncommit.reason, "non_commitment_action")
        self.assertEqual(no_preference.status, DecisionStatus.ACCEPT)
        self.assertEqual(no_preference.reason, "no_explicit_preference")

    def test_full_guard_accepts_winner_rejects_dominated_and_infeasible(self):
        winner, dominated, infeasible, ledger = price_fixture()
        guard = ChoiceIntegrityGuard()

        accepted = guard.certify(
            "click[Buy Now]",
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
        )
        rejected = guard.certify(
            "click[Buy Now]",
            selected_candidate=dominated,
            ledger=ledger,
            preference=Preference.price_min(),
        )
        hard_rejected = guard.certify(
            "click[Buy Now]",
            selected_candidate=infeasible,
            ledger=ledger,
            preference=Preference.price_min(),
        )

        self.assertEqual(accepted.status, DecisionStatus.ACCEPT)
        self.assertEqual(rejected.status, DecisionStatus.REJECT)
        self.assertEqual(
            rejected.dominating_candidate_ids, (winner.candidate_id,)
        )
        self.assertEqual(hard_rejected.status, DecisionStatus.REJECT)
        self.assertEqual(hard_rejected.reason, "hard_infeasible_candidate")

    def test_missing_evidence_or_coverage_defers(self):
        item = candidate("A", price=10)
        incomplete = CandidateLedger.from_candidates(
            (item,),
            source="results_page",
            comparison_complete=False,
        )
        missing_price_item = candidate("B", price=None)
        missing_price = CandidateLedger.from_candidates(
            (missing_price_item,),
            source="results_page",
        )
        guard = ChoiceIntegrityGuard()

        for selected, ledger in (
            (item, incomplete),
            (missing_price_item, missing_price),
        ):
            with self.subTest(selected=selected.asin):
                decision = guard.certify(
                    "click[Buy Now]",
                    selected_candidate=selected,
                    ledger=ledger,
                    preference=Preference.price_min(),
                )
                self.assertEqual(decision.status, DecisionStatus.DEFER)
                self.assertEqual(
                    decision.reason, "comparison_evidence_incomplete"
                )

    def test_ablation_contracts_preserve_hard_feasibility(self):
        _, dominated, infeasible, ledger = price_fixture()
        no_dominance = ChoiceIntegrityGuard(ablation="no_dominance")
        self.assertEqual(
            no_dominance.certify(
                "click[Buy Now]",
                selected_candidate=dominated,
                ledger=ledger,
                preference=Preference.price_min(),
            ).status,
            DecisionStatus.ACCEPT,
        )
        self.assertEqual(
            no_dominance.certify(
                "click[Buy Now]",
                selected_candidate=infeasible,
                ledger=ledger,
                preference=Preference.price_min(),
            ).status,
            DecisionStatus.REJECT,
        )

        no_ledger = ChoiceIntegrityGuard(ablation="no_ledger")
        self.assertEqual(
            no_ledger.certify(
                "click[Buy Now]",
                selected_candidate=dominated,
                ledger=None,
                preference=Preference.price_min(),
            ).status,
            DecisionStatus.ACCEPT,
        )
        self.assertEqual(
            no_ledger.certify(
                "click[Buy Now]",
                selected_candidate=dominated.candidate_id,
                ledger=None,
                preference=Preference.price_min(),
            ).status,
            DecisionStatus.DEFER,
        )
        self.assertEqual(
            no_ledger.certify(
                "click[Buy Now]",
                selected_candidate=infeasible,
                ledger=None,
                preference=Preference.price_min(),
            ).status,
            DecisionStatus.REJECT,
        )

    def test_ablations_fail_closed_on_conflicting_feasibility(self):
        item = candidate("A", feasible=True, price=10)
        ledger = CandidateLedger.from_candidates(
            (item,),
            source="results_page",
        )
        entry = ledger.upsert(
            candidate("A", feasible=False, price=10),
            source="product_page",
        )

        no_dominance = ChoiceIntegrityGuard(ablation="no_dominance")
        no_ledger = ChoiceIntegrityGuard(ablation="no_ledger")

        self.assertEqual(
            no_dominance.certify(
                "click[Buy Now]",
                selected_candidate=entry,
                ledger=ledger,
                preference=Preference.price_min(),
            ).status,
            DecisionStatus.DEFER,
        )
        self.assertEqual(
            no_ledger.certify(
                "click[Buy Now]",
                selected_candidate=entry,
                ledger=None,
                preference=Preference.price_min(),
            ).status,
            DecisionStatus.DEFER,
        )


class DeterministicChoicePlannerTests(unittest.TestCase):
    def setUp(self):
        self.winner, self.dominated, self.infeasible, self.ledger = (
            price_fixture()
        )
        self.planner = DeterministicChoicePlanner()

    def test_opens_public_ledger_winner_not_infeasible_cheapest(self):
        plan = self.planner.plan(
            ledger=self.ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("WINNER", "CHEAP-BUT-WRONG"),
            },
            canonical_query="blue shirt",
        )

        self.assertEqual(plan.action, "click[WINNER]")
        self.assertEqual(plan.target_candidate_id, self.winner.candidate_id)

    def test_selects_options_before_buying(self):
        select = self.planner.plan(
            ledger=self.ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("blue", "M", "Buy Now"),
            },
            canonical_query="blue shirt",
            current_candidate=self.winner,
            selected_options={"color": "blue"},
        )
        buy = self.planner.plan(
            ledger=self.ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("Buy Now",),
            },
            canonical_query="blue shirt",
            current_candidate=self.winner,
            selected_options={"color": "blue", "size": "M"},
        )

        self.assertEqual(select.action, "click[M]")
        self.assertEqual(buy.action, "click[Buy Now]")

    def test_never_buys_when_required_option_is_unselected_and_unavailable(self):
        plan = self.planner.plan(
            ledger=self.ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("Buy Now",),
            },
            canonical_query="blue shirt",
            current_candidate=self.winner,
            selected_options={"color": "blue"},
        )

        self.assertIsNone(plan.action)
        self.assertFalse(plan.legal)

    def test_incomplete_ledger_uses_only_legal_information_gathering(self):
        incomplete = CandidateLedger.from_candidates(
            (self.winner,),
            source="results_page",
            comparison_complete=False,
        )
        search = self.planner.plan(
            ledger=incomplete,
            preference=Preference.price_min(),
            available_actions={"has_search_bar": True, "clickables": ()},
            canonical_query="blue shirt",
        )
        back = self.planner.plan(
            ledger=incomplete,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("Back to Search",),
            },
            canonical_query="blue shirt",
        )
        blocked = self.planner.plan(
            ledger=incomplete,
            preference=Preference.price_min(),
            available_actions={"has_search_bar": False, "clickables": ()},
            canonical_query="blue shirt",
        )

        self.assertEqual(search.action, "search[blue shirt]")
        self.assertTrue(AvailableActions.parse(
            {"has_search_bar": True}
        ).legal_action(search.action))
        self.assertEqual(back.action, "click[Back to Search]")
        self.assertIsNone(blocked.action)
        self.assertFalse(blocked.legal)


class ChoiceIntegrityDefenseTests(unittest.TestCase):
    def test_hard_goal_rejection_has_priority_over_preference_winner(self):
        winner, _, _, ledger = price_fixture()
        result = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=False,
            goal_repair=None,
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ("Buy Now",)},
            canonical_query="blue shirt",
            hard_constraints={"color": "blue"},
        )

        self.assertIsNone(result.executed_action)
        self.assertEqual(result.choice.reason, "hard_goal_rejected")

    def test_dominated_commitment_is_repaired_toward_ledger_winner(self):
        winner, dominated, _, ledger = price_fixture()
        result = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=dominated,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("WINNER", "Buy Now"),
            },
            canonical_query="blue shirt",
            repair_certifier=lambda _action: True,
        )

        self.assertEqual(result.executed_action, "click[WINNER]")
        self.assertTrue(result.intervened)
        self.assertEqual(result.plan.target_candidate_id, winner.candidate_id)

    def test_choice_planner_repair_requires_external_full_certificate(self):
        winner, dominated, _, ledger = price_fixture()
        missing_certifier = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=dominated,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("WINNER", "Buy Now"),
            },
            canonical_query="blue shirt",
        )
        denied_certifier = mock.Mock(return_value=False)
        blocked = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=dominated,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("WINNER", "Buy Now"),
            },
            canonical_query="blue shirt",
            repair_certifier=denied_certifier,
        )
        accepted_certifier = mock.Mock(return_value=True)
        accepted = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=dominated,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={
                "has_search_bar": False,
                "clickables": ("WINNER", "Buy Now"),
            },
            canonical_query="blue shirt",
            repair_certifier=accepted_certifier,
        )

        self.assertIsNone(missing_certifier.executed_action)
        self.assertFalse(missing_certifier.repair_certified)
        self.assertIsNone(blocked.executed_action)
        self.assertFalse(blocked.repair_certified)
        self.assertEqual(
            blocked.reason,
            "choice_rejected_and_uncertified_repair_blocked",
        )
        denied_certifier.assert_called_once_with("click[WINNER]")
        self.assertEqual(accepted.executed_action, "click[WINNER]")
        self.assertTrue(accepted.repair_certified)
        accepted_certifier.assert_called_once_with("click[WINNER]")

    def test_only_a_legal_hard_goal_repair_can_continue(self):
        winner, _, _, ledger = price_fixture()
        legal = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=False,
            goal_repair="click[Back to Search]",
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ("Back to Search",)},
            canonical_query="blue shirt",
            repair_certifier=lambda _action: True,
        )
        illegal = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=False,
            goal_repair="click[Secret]",
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ("Back to Search",)},
            canonical_query="blue shirt",
        )
        uncertified = ChoiceIntegrityDefense().intercept(
            "click[Buy Now]",
            goal_accepted=False,
            goal_repair="click[Back to Search]",
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ("Back to Search",)},
            canonical_query="blue shirt",
            repair_certifier=lambda _action: False,
        )

        self.assertEqual(legal.executed_action, "click[Back to Search]")
        self.assertIsNone(illegal.executed_action)
        self.assertIsNone(uncertified.executed_action)
        self.assertEqual(
            uncertified.reason,
            "hard_goal_rejected_without_certified_repair",
        )


if __name__ == "__main__":
    unittest.main()
