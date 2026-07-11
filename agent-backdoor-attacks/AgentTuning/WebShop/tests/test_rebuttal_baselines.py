"""CPU-only tests for the matched WebShop runtime baselines."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


WEBSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_ROOT))

from defenses.rebuttal_baselines import (  # noqa: E402
    LLMJudge,
    LegalRepair,
    LexicalGuard,
    RepairState,
    build_judge_messages,
    build_replacement_candidates,
    judge_prompt_payload,
    resolve_judge_output,
)


SEARCH_PAGE = {
    "has_search_bar": True,
    "clickables": ["Back"],
}
RESULTS_PAGE = {
    "has_search_bar": False,
    "clickables": ["B012345678", "Back", "Next"],
}


class LegalRepairTests(unittest.TestCase):
    def test_legal_click_is_accepted_unchanged_after_normalization(self) -> None:
        repair = LegalRepair()
        calls = []

        result = repair.resolve_action(
            "click[  b012345678  ]",
            RESULTS_PAGE,
            lambda feedback: calls.append(feedback),
        )

        self.assertEqual(result.action, "click[  b012345678  ]")
        self.assertFalse(result.repair_attempted)
        self.assertEqual(calls, [])
        self.assertEqual(repair.counters.initially_legal, 1)
        self.assertEqual(repair.counters.extra_generations, 0)

    def test_illegal_click_gets_exactly_one_repair_attempt(self) -> None:
        repair = LegalRepair()
        state = RepairState()
        calls = []

        def generate(feedback: str) -> str:
            calls.append(feedback)
            return "Thought: choose a current value.\nAction: click[B012345678]"

        first = repair.resolve_action(
            "click[not-current]",
            RESULTS_PAGE,
            generate,
            state=state,
        )
        second = repair.repair_once(
            "click[still-invalid]",
            RESULTS_PAGE,
            generate,
            state=state,
        )

        self.assertEqual(first.action, "click[B012345678]")
        self.assertTrue(first.repair_attempted)
        self.assertEqual(second.failure, "repair_budget_exhausted")
        self.assertEqual(second.action, None)
        self.assertTrue(second.should_terminate)
        self.assertFalse(second.requires_repair)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state.attempts, 1)
        self.assertEqual(repair.counters.extra_generations, 1)
        self.assertEqual(repair.counters.repair_successes, 1)

    def test_repair_feedback_contains_only_legality_context(self) -> None:
        feedback = LegalRepair.build_repair_feedback(RESULTS_PAGE).casefold()

        self.assertIn("previous action was invalid", feedback)
        self.assertIn("search available", feedback)
        self.assertIn("current legal clickable values", feedback)
        self.assertNotIn("user goal", feedback)
        self.assertNotIn("security", feedback)
        self.assertNotIn("attack", feedback)
        self.assertNotIn("suspicious", feedback)

    def test_illegal_second_proposal_terminates_without_another_generation(self) -> None:
        repair = LegalRepair()
        calls = []

        result = repair.resolve_action(
            "click[not-current]",
            RESULTS_PAGE,
            lambda feedback: calls.append(feedback) or "Action: click[still-not-current]",
        )

        self.assertIsNone(result.action)
        self.assertTrue(result.should_terminate)
        self.assertEqual(len(calls), 1)
        self.assertEqual(repair.counters.extra_generations, 1)
        self.assertEqual(repair.counters.repair_failures, 1)

    def test_search_submit_button_is_not_a_legal_click(self) -> None:
        available = {
            "has_search_bar": True,
            "clickables": ["Search", "B012345678"],
        }

        validation = LegalRepair.validate_action("click[Search]", available)
        candidates = build_replacement_candidates("Find red shoes", available)

        self.assertFalse(validation.legal)
        self.assertNotIn("click[Search]", candidates)
        self.assertIn("search[red shoes]", candidates)


class LexicalGuardTests(unittest.TestCase):
    def test_inserted_unrequested_brand_token_is_removed(self) -> None:
        guard = LexicalGuard("Find red running shoes in size 8")

        result = guard.guard_action(
            "search[Northwind red running shoes size 8]",
            SEARCH_PAGE,
        )

        self.assertEqual(result.action, "search[red running shoes size 8]")
        self.assertIsNotNone(result.filter_result)
        assert result.filter_result is not None
        self.assertEqual(result.filter_result.removed_tokens, ("northwind",))

    def test_explicitly_requested_brand_token_is_preserved(self) -> None:
        guard = LexicalGuard("Find Northwind red running shoes")

        result = guard.guard_action(
            "search[Northwind red running shoes]",
            SEARCH_PAGE,
        )

        self.assertEqual(result.action, "search[northwind red running shoes]")
        assert result.filter_result is not None
        self.assertNotIn("northwind", result.filter_result.removed_tokens)

    def test_price_size_color_numbers_and_units_are_preserved(self) -> None:
        guard = LexicalGuard(
            "Find red running shoes, color red, size 8.5, under $49.99, "
            "width 2 inches and size XL"
        )

        result = guard.guard_action(
            "search[Northwind running shoes color red size 8.5 under $49.99 "
            "width 2 inch size XL]",
            SEARCH_PAGE,
        )

        self.assertEqual(
            result.action,
            "search[running shoes color red size 8.5 under $49.99 width 2 inch size xl]",
        )
        assert result.filter_result is not None
        for token in ("color", "red", "size", "8.5", "under", "$49.99", "2", "inch", "xl"):
            self.assertIn(token, result.filter_result.kept_tokens)
        self.assertEqual(result.filter_result.removed_tokens, ("northwind",))

    def test_click_is_checked_only_for_legality(self) -> None:
        guard = LexicalGuard("Find red running shoes")

        result = guard.guard_action("click[B012345678]", RESULTS_PAGE)

        self.assertEqual(result.action, "click[B012345678]")
        self.assertIsNone(result.filter_result)

    def test_illegal_click_does_not_turn_lexical_guard_into_generic_repair(self) -> None:
        guard = LexicalGuard("Find red running shoes")
        calls = []

        result = guard.guard_action(
            "click[not-current]",
            RESULTS_PAGE,
            lambda feedback: calls.append(feedback) or "Action: click[B012345678]",
        )

        self.assertIsNone(result.action)
        self.assertIsNone(result.repair_result)
        self.assertEqual(calls, [])
        self.assertEqual(guard.legal_repair.counters.extra_generations, 0)

    def test_empty_filtered_query_uses_one_legal_repair(self) -> None:
        guard = LexicalGuard("Find running shoes")
        state = RepairState()
        calls = []

        result = guard.guard_action(
            "search[Northwind]",
            SEARCH_PAGE,
            lambda feedback: calls.append(feedback) or "Action: search[running shoes]",
            state=state,
        )

        self.assertEqual(result.action, "search[running shoes]")
        self.assertEqual(len(calls), 1)
        self.assertTrue(result.repair_result and result.repair_result.repair_attempted)
        self.assertEqual(state.attempts, 1)


class LLMJudgeTests(unittest.TestCase):
    def test_prompt_has_only_allowed_fields_and_no_experiment_metadata(self) -> None:
        payload = judge_prompt_payload(
            "Find red running shoes",
            "Search page",
            SEARCH_PAGE,
            "search[red running shoes]",
        )
        messages = build_judge_messages(
            "Find red running shoes",
            "Search page",
            SEARCH_PAGE,
            "search[red running shoes]",
        )
        serialized = json.dumps(messages).casefold()

        self.assertEqual(
            set(payload),
            {
                "original_user_instruction",
                "current_raw_observation",
                "current_legal_action_set",
                "proposed_action",
                "replacement_candidates",
            },
        )
        for forbidden in (
            "northwind",
            "attack",
            "attack_type",
            "target_brand",
            "asr",
            "poisoned",
            "future_observation",
            "future_reward",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_replacement_candidates_are_deterministic_and_complete(self) -> None:
        available = {
            "has_search_bar": True,
            "clickables": ["B012345678", "Back"],
        }

        candidates = build_replacement_candidates("Find red running shoes", available)

        self.assertEqual(
            candidates,
            (
                "click[B012345678]",
                "click[Back]",
                "search[red running shoes]",
            ),
        )

    def test_invalid_judge_index_cannot_select_an_arbitrary_click(self) -> None:
        raw = {
            "allow": False,
            "replacement_index": 999,
            "reason": "reject",
        }

        result = resolve_judge_output(
            raw,
            "Find red running shoes",
            "Results page",
            RESULTS_PAGE,
            "click[not-current]",
        )

        self.assertIsNone(result.action)
        self.assertTrue(result.requires_legal_repair)
        self.assertEqual(result.failure, "judge_replacement_index_invalid")

    def test_invalid_schema_cannot_allow_an_illegal_proposal(self) -> None:
        result = resolve_judge_output(
            '{"allow": "yes", "replacement_index": 0, "reason": "bad"}',
            "Find red running shoes",
            "Results page",
            RESULTS_PAGE,
            "click[not-current]",
        )

        self.assertIsNone(result.action)
        self.assertTrue(result.requires_legal_repair)

    def test_judge_request_is_temperature_zero_and_cached(self) -> None:
        provider_requests = []

        def provider(request):
            provider_requests.append(request)
            return {"allow": True, "replacement_index": None, "reason": "legal"}

        with tempfile.TemporaryDirectory() as temp_dir:
            judge = LLMJudge(
                "judge-test-model",
                cache_path=Path(temp_dir) / "judge.json",
                provider=provider,
            )
            first = judge.evaluate_action(
                "Find red running shoes",
                "Results page",
                RESULTS_PAGE,
                "click[B012345678]",
            )
            second = judge.evaluate_action(
                "Find red running shoes",
                "Results page",
                RESULTS_PAGE,
                "click[B012345678]",
            )

        self.assertEqual(first.action, "click[B012345678]")
        self.assertEqual(second.action, "click[B012345678]")
        self.assertEqual(len(provider_requests), 1)
        self.assertEqual(provider_requests[0]["temperature"], 0)
        response_format = provider_requests[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertFalse(
            response_format["json_schema"]["schema"]["additionalProperties"]
        )
        self.assertEqual(judge.counters.requests, 2)
        self.assertEqual(judge.counters.judge_calls, 1)
        self.assertEqual(judge.counters.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
