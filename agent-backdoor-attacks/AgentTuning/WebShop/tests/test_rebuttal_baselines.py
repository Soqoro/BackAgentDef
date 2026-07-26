"""CPU-only tests for the matched WebShop runtime baselines."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


WEBSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_ROOT))

from defenses.rebuttal_baselines import (  # noqa: E402
    LLMPricing,
    LLMJudge,
    LLMUsage,
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
                pricing=LLMPricing(
                    input_usd_per_million=2.0,
                    output_usd_per_million=8.0,
                ),
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
        self.assertEqual(provider_requests[0]["service_tier"], "default")
        response_format = provider_requests[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertFalse(
            response_format["json_schema"]["schema"]["additionalProperties"]
        )
        self.assertEqual(judge.counters.requests, 2)
        self.assertEqual(judge.counters.judge_calls, 1)
        self.assertEqual(judge.counters.api_calls, 1)
        self.assertEqual(judge.counters.cache_hits, 1)
        self.assertEqual(judge.counters.usage_reported_call_count, 0)
        self.assertEqual(judge.counters.usage_missing_call_count, 1)
        self.assertIsNone(first.estimated_cost_usd)
        self.assertEqual(second.estimated_cost_usd, 0.0)
        self.assertFalse(first.llm_usage.usage_reported)
        self.assertEqual(first.action, second.action)
        # The decision is backward-compatible, but absent usage makes aggregate
        # spend unknown rather than silently reporting a zero-cost provider call.
        self.assertIsNone(judge.counters.estimated_cost_usd)

    def test_full_object_response_tracks_exact_usage_cost_and_cache(self) -> None:
        decision = json.dumps(
            {"allow": True, "replacement_index": None, "reason": "legal"}
        )
        response = SimpleNamespace(
            model="judge-actual-2026-07-01",
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=250),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=50),
            ),
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=decision))
            ],
        )
        calls = []
        pricing = LLMPricing(
            input_usd_per_million=2.0,
            cached_input_usd_per_million=0.5,
            output_usd_per_million=8.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            judge = LLMJudge(
                "judge-requested-alias",
                cache_path=Path(temp_dir) / "judge.json",
                provider=lambda request: calls.append(request) or response,
                pricing=pricing,
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

        expected_cost = 0.003225
        self.assertEqual(len(calls), 1)
        self.assertEqual(first.action, "click[B012345678]")
        self.assertEqual(first.llm_usage.input_tokens, 1000)
        self.assertEqual(first.llm_usage.output_tokens, 200)
        self.assertEqual(first.llm_usage.total_tokens, 1200)
        self.assertEqual(first.llm_usage.cached_input_tokens, 250)
        self.assertEqual(first.llm_usage.reasoning_tokens, 50)
        self.assertEqual(first.llm_usage.model, "judge-actual-2026-07-01")
        self.assertAlmostEqual(first.estimated_cost_usd or 0.0, expected_cost)
        self.assertEqual(
            second.llm_usage,
            LLMUsage(usage_reported=False),
        )
        self.assertEqual(second.estimated_cost_usd, 0.0)
        self.assertEqual(judge.counters.input_tokens, 1000)
        self.assertEqual(judge.counters.output_tokens, 200)
        self.assertEqual(judge.counters.total_tokens, 1200)
        self.assertEqual(judge.counters.cached_input_tokens, 250)
        self.assertEqual(judge.counters.reasoning_tokens, 50)
        self.assertEqual(
            judge.counters.actual_models,
            ["judge-actual-2026-07-01"],
        )
        self.assertEqual(judge.counters.usage_reported_call_count, 1)
        self.assertEqual(judge.counters.usage_missing_call_count, 0)
        self.assertAlmostEqual(
            judge.counters.estimated_cost_usd or 0.0,
            expected_cost,
        )
        self.assertAlmostEqual(
            judge.counters.known_estimated_cost_usd or 0.0,
            expected_cost,
        )
        result_log = first.to_dict()
        self.assertEqual(result_log["llm_usage"]["model"], "judge-actual-2026-07-01")
        self.assertAlmostEqual(result_log["estimated_cost_usd"], expected_cost)

    def test_full_mapping_response_extracts_decision_and_usage(self) -> None:
        response = {
            "model": "judge-mapping-model",
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": 22,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "allow": False,
                                "replacement_index": 0,
                                "reason": "replace",
                            }
                        )
                    }
                }
            ],
        }
        judge = LLMJudge(
            "judge-requested-alias",
            provider=lambda request: response,
            pricing=LLMPricing(
                input_usd_per_million=1.0,
                cached_input_usd_per_million=0.5,
                output_usd_per_million=3.0,
            ),
        )

        result = judge.evaluate_action(
            "Find red running shoes",
            "Results page",
            RESULTS_PAGE,
            "click[not-current]",
        )

        self.assertEqual(result.action, "click[B012345678]")
        self.assertTrue(result.replacement_applied)
        self.assertEqual(result.llm_usage.model, "judge-mapping-model")
        self.assertEqual(result.llm_usage.input_tokens, 17)
        self.assertEqual(result.llm_usage.output_tokens, 5)
        self.assertEqual(result.llm_usage.cached_input_tokens, 3)
        self.assertEqual(result.llm_usage.reasoning_tokens, 2)
        self.assertAlmostEqual(result.estimated_cost_usd or 0.0, 0.0000305)

    def test_invalid_output_still_accounts_usage_and_is_not_cached(self) -> None:
        response = {
            "model": "judge-invalid-output-model",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
            "choices": [{"message": {"content": "not-json"}}],
        }
        calls = []
        judge = LLMJudge(
            "judge-requested-alias",
            provider=lambda request: calls.append(request) or response,
            pricing=LLMPricing(
                input_usd_per_million=2.0,
                cached_input_usd_per_million=1.0,
                output_usd_per_million=4.0,
            ),
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

        self.assertEqual(len(calls), 2)
        self.assertFalse(first.cache_hit)
        self.assertFalse(second.cache_hit)
        self.assertTrue((first.failure or "").startswith("invalid_json:"))
        self.assertEqual(first.llm_usage.total_tokens, 12)
        self.assertAlmostEqual(first.estimated_cost_usd or 0.0, 0.000026)
        self.assertEqual(judge.counters.requests, 2)
        self.assertEqual(judge.counters.judge_calls, 2)
        self.assertEqual(judge.counters.cache_hits, 0)
        self.assertEqual(judge.counters.usage_reported_call_count, 2)
        self.assertEqual(judge.counters.input_tokens, 20)
        self.assertEqual(judge.counters.output_tokens, 4)
        self.assertEqual(judge.counters.total_tokens, 24)
        self.assertAlmostEqual(
            judge.counters.estimated_cost_usd or 0.0,
            0.000052,
        )

    def test_raw_string_provider_stays_compatible_without_partial_cost(self) -> None:
        decision = json.dumps(
            {"allow": True, "replacement_index": None, "reason": "legal"}
        )
        responses = iter(
            (
                {
                    "model": "judge-reported-model",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                    "choices": [{"message": {"content": decision}}],
                },
                decision,
            )
        )
        judge = LLMJudge(
            "judge-model",
            provider=lambda request: next(responses),
            pricing=LLMPricing(
                input_usd_per_million=1.0,
                output_usd_per_million=2.0,
            ),
        )

        reported = judge.evaluate_action(
            "Find red running shoes",
            "Results page",
            RESULTS_PAGE,
            "click[B012345678]",
        )
        raw_string = judge.evaluate_action(
            "Find red running shoes",
            "Results page",
            RESULTS_PAGE,
            "click[Back]",
        )

        self.assertEqual(reported.action, "click[B012345678]")
        self.assertEqual(raw_string.action, "click[Back]")
        self.assertFalse(raw_string.llm_usage.usage_reported)
        self.assertEqual(judge.counters.usage_reported_call_count, 1)
        self.assertEqual(judge.counters.usage_missing_call_count, 1)
        self.assertAlmostEqual(
            judge.counters.known_estimated_cost_usd or 0.0,
            0.00012,
        )
        self.assertIsNone(judge.counters.estimated_cost_usd)

    def test_malformed_usage_does_not_change_a_valid_judge_decision(self) -> None:
        response = {
            "model": "judge-bad-usage-model",
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": 3,
                "total_tokens": 2,
            },
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "allow": True,
                                "replacement_index": None,
                                "reason": "legal",
                            }
                        )
                    }
                }
            ],
        }
        judge = LLMJudge(
            "judge-model",
            provider=lambda request: response,
            pricing=LLMPricing(
                input_usd_per_million=1.0,
                output_usd_per_million=2.0,
            ),
        )

        result = judge.evaluate_action(
            "Find red running shoes",
            "Results page",
            RESULTS_PAGE,
            "click[B012345678]",
        )

        self.assertEqual(result.action, "click[B012345678]")
        self.assertIsNone(result.failure)
        self.assertFalse(result.llm_usage.usage_reported)
        self.assertEqual(result.llm_usage.model, "judge-bad-usage-model")
        self.assertEqual(judge.counters.usage_missing_call_count, 1)
        self.assertIsNone(judge.counters.estimated_cost_usd)

    def test_default_provider_returns_full_response(self) -> None:
        response = object()
        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = lambda timeout: SimpleNamespace(  # type: ignore[attr-defined]
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **request: response,
                )
            )
        )
        judge = LLMJudge("judge-model")

        with patch.dict(sys.modules, {"openai": fake_openai}):
            returned = judge._default_provider({"model": "judge-model"})

        self.assertIs(returned, response)


if __name__ == "__main__":
    unittest.main()
