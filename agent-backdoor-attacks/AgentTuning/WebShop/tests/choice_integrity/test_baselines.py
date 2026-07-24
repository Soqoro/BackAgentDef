import json
import types
import unittest

from choice_integrity.baselines import (
    CallableStateAwareVerifier,
    DeterministicDominanceVerifier,
    DeterministicVerifierStub,
    GoalOnlyBaseline,
    OpenAIStateAwareVerifier,
    StateAwareVerifierDefense,
    UndefendedBaseline,
    VerifierRequest,
    VerifierStatus,
)
from choice_integrity.defense import DecisionStatus
from choice_integrity.ledger import CandidateLedger
from choice_integrity.schema import Candidate, Preference


def candidate(asin, *, feasible=True, price=None, rating=None, metadata=None):
    return Candidate(
        asin=asin,
        feasible=feasible,
        price=price,
        rating=rating,
        evidence={"price": {"value": price}, "rating": {"value": rating}},
        metadata=metadata or {},
    )


def fixture():
    winner = candidate(
        "WINNER",
        price=10,
        rating=4.8,
        metadata={"winner_ids": ["DO-NOT-LEAK"]},
    )
    dominated = candidate(
        "DOMINATED",
        price=20,
        rating=4.2,
        metadata={"attacker_target_ids": ["DO-NOT-LEAK"]},
    )
    ledger = CandidateLedger.from_candidates(
        (winner, dominated),
        source="fixed_shortlist",
    )
    return winner, dominated, ledger


def request_for(selected, candidates, preference=None):
    return VerifierRequest(
        action="click[Buy Now]",
        selected_candidate_id=selected.candidate_id,
        preference=preference or Preference.price_min(),
        candidates=tuple(candidates),
        canonical_query="watch",
    )


class StateAwareVerifierTests(unittest.TestCase):
    def test_deterministic_verifier_accepts_winner_and_rejects_dominated(self):
        winner, dominated, _ = fixture()
        verifier = DeterministicDominanceVerifier()

        self.assertEqual(
            verifier.verify(request_for(winner, (winner, dominated))).status,
            VerifierStatus.ACCEPT,
        )
        self.assertEqual(
            verifier.verify(request_for(dominated, (winner, dominated))).status,
            VerifierStatus.REJECT,
        )

    def test_deterministic_verifier_defers_on_missing_comparison_evidence(self):
        selected = candidate("A", price=10, rating=None)
        other = candidate("B", price=12, rating=4.2)
        verdict = DeterministicDominanceVerifier().verify(
            request_for(
                selected,
                (selected, other),
                Preference.rating_max_price_tiebreak(),
            )
        )

        self.assertEqual(verdict.status, VerifierStatus.DEFER)

    def test_callable_adapter_fails_closed_on_errors_and_malformed_values(self):
        winner, dominated, _ = fixture()
        request = request_for(winner, (winner, dominated))

        def fail(_):
            raise RuntimeError("offline")

        failed = CallableStateAwareVerifier(fail).verify(request)
        malformed = CallableStateAwareVerifier(
            lambda _: {"decision": "maybe", "reason": "guess"}
        ).verify(request)

        self.assertEqual(failed.status, VerifierStatus.DEFER)
        self.assertEqual(malformed.status, VerifierStatus.DEFER)

    def test_optional_openai_adapter_fails_closed_without_network(self):
        winner, dominated, _ = fixture()
        request = request_for(winner, (winner, dominated))

        class Completions:
            def __init__(self, content=None, error=None):
                self.content = content
                self.error = error

            def create(self, **_):
                if self.error is not None:
                    raise self.error
                message = types.SimpleNamespace(content=self.content)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        malformed_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=Completions('{"decision":"accept"}')
            )
        )
        failing_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=Completions(error=RuntimeError("offline"))
            )
        )

        malformed = OpenAIStateAwareVerifier(
            model="test-model", client=malformed_client
        ).verify(request)
        failed = OpenAIStateAwareVerifier(
            model="test-model", client=failing_client
        ).verify(request)

        self.assertEqual(malformed.status, VerifierStatus.DEFER)
        self.assertEqual(failed.status, VerifierStatus.DEFER)


class StateAwareVerifierDefenseTests(unittest.TestCase):
    def test_verifier_runs_only_at_commitment(self):
        winner, _, ledger = fixture()
        verifier = DeterministicVerifierStub(VerifierStatus.ACCEPT)
        defense = StateAwareVerifierDefense(verifier)

        result = defense.intercept(
            "click[WINNER]",
            goal_accepted=True,
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ("WINNER",)},
            canonical_query="watch",
        )

        self.assertEqual(verifier.calls, 0)
        self.assertEqual(result.executed_action, "click[WINNER]")

    def test_hard_infeasible_preflight_stops_verifier_and_fails_closed(self):
        winner, _, ledger = fixture()
        impossible = candidate("IMPOSSIBLE", feasible=False, price=1)
        ledger.upsert(impossible, source="fixed_shortlist")
        verifier = DeterministicVerifierStub(VerifierStatus.ACCEPT)
        defense = StateAwareVerifierDefense(verifier)

        result = defense.intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=impossible,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ()},
            canonical_query="watch",
        )

        self.assertEqual(verifier.calls, 0)
        self.assertEqual(result.choice.status, DecisionStatus.REJECT)
        self.assertIsNone(result.executed_action)
        self.assertEqual(result.plan.target_candidate_id, winner.candidate_id)

    def test_verifier_defer_uses_shared_deterministic_repair(self):
        winner, dominated, ledger = fixture()
        verifier = DeterministicVerifierStub(
            VerifierStatus.DEFER, "uncertain"
        )
        result = StateAwareVerifierDefense(verifier).intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=dominated,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ("WINNER",)},
            canonical_query="watch",
        )

        self.assertEqual(result.executed_action, "click[WINNER]")
        self.assertEqual(result.plan.target_candidate_id, winner.candidate_id)

    def test_verifier_request_contains_no_evaluator_labels(self):
        winner, dominated, ledger = fixture()
        verifier = DeterministicVerifierStub(VerifierStatus.ACCEPT)
        StateAwareVerifierDefense(verifier).intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ("Buy Now",)},
            canonical_query="watch",
            hard_constraints={"category": "watch"},
        )

        payload = json.dumps(verifier.last_request.to_dict()).lower()
        self.assertNotIn("winner_ids", payload)
        self.assertNotIn("attacker_target", payload)
        self.assertNotIn("do-not-leak", payload)
        self.assertEqual(
            dict(verifier.last_request.candidates[0].metadata),
            {},
        )

    def test_malformed_verifier_result_cannot_accept(self):
        winner, _, ledger = fixture()

        class MalformedVerifier:
            def verify(self, _):
                return {"status": "accept", "reason": "not typed"}

        result = StateAwareVerifierDefense(MalformedVerifier()).intercept(
            "click[Buy Now]",
            goal_accepted=True,
            selected_candidate=winner,
            ledger=ledger,
            preference=Preference.price_min(),
            available_actions={"clickables": ()},
            canonical_query="watch",
        )

        self.assertEqual(result.choice.status, DecisionStatus.DEFER)
        self.assertIsNone(result.executed_action)


class SimpleBaselineTests(unittest.TestCase):
    def test_undefended_returns_proposal_and_goal_only_checks_legal_repair(self):
        self.assertEqual(
            UndefendedBaseline().intercept("click[Buy Now]"),
            "click[Buy Now]",
        )
        goal_only = GoalOnlyBaseline()
        self.assertEqual(
            goal_only.intercept(
                "click[Buy Now]",
                goal_accepted=False,
                goal_repair="click[Back to Search]",
                available_actions={"clickables": ("Back to Search",)},
            ),
            "click[Back to Search]",
        )
        self.assertIsNone(
            goal_only.intercept(
                "click[Buy Now]",
                goal_accepted=False,
                goal_repair="click[Secret]",
                available_actions={"clickables": ("Back to Search",)},
            )
        )


if __name__ == "__main__":
    unittest.main()
