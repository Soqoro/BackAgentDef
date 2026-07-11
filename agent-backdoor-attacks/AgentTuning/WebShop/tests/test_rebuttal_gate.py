"""CPU-only tests for parser contracts, explicit GATE modes, and ASR denominators."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


WEBSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_ROOT))

from defenses.gate import GateDefense  # noqa: E402
from defenses.goal_contract import (  # noqa: E402
    GoalContract,
    GoalContractExtraction,
    GoalContractParseError,
    goal_contract_cache_key,
)
from defenses.rebuttal_metrics import attack_metric_summaries  # noqa: E402


class GoalParserContractTests(unittest.TestCase):
    def test_fail_fast_rejects_missing_api_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GoalContractParseError):
                GoalContractExtraction(
                    use_openai=True,
                    openai_model="paper-parser",
                    require_success=True,
                )

    def test_cache_key_uses_exact_instruction_and_model(self) -> None:
        base = goal_contract_cache_key("Find red shoes", "model-a")
        self.assertNotEqual(base, goal_contract_cache_key("Find  red shoes", "model-a"))
        self.assertNotEqual(base, goal_contract_cache_key("Find red shoes", "model-b"))

    def test_fail_fast_rejects_parser_call_fallback(self) -> None:
        failed = GoalContract(
            raw_query="Find red shoes",
            intent="Find red shoes",
            extractor="regex_fallback",
            extraction_error="simulated provider failure",
        )
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            extraction = GoalContractExtraction(
                use_openai=True,
                openai_model="paper-parser",
                require_success=True,
            )
            extraction.openai_extractor.extract = mock.Mock(return_value=failed)
            with self.assertRaisesRegex(GoalContractParseError, "simulated provider failure"):
                extraction.extract("Find red shoes")

        self.assertEqual(extraction.parser_error_count, 1)
        self.assertEqual(extraction.parser_fallback_count, 1)

    def test_successful_contract_is_reused_from_cache(self) -> None:
        instruction = "Find red shoes under $50"
        contract = GoalContract(
            raw_query=instruction,
            intent="find shoes",
            positive_constraints=["red", "under $50"],
            extractor="openai_goal_contract",
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "goals.json"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                first = GoalContractExtraction(
                    use_openai=True,
                    openai_model="paper-parser",
                    require_success=True,
                    cache_path=str(cache),
                )
                first.openai_extractor.extract = mock.Mock(return_value=contract)
                self.assertEqual(first.extract(instruction).intent, "find shoes")

                second = GoalContractExtraction(
                    use_openai=True,
                    openai_model="paper-parser",
                    require_success=True,
                    cache_path=str(cache),
                )
                second.openai_extractor.extract = mock.Mock(
                    side_effect=AssertionError("cache miss")
                )
                loaded = second.extract(instruction)

            self.assertEqual(loaded.positive_constraints, ["red", "under $50"])
            self.assertEqual(second.parser_cache_hits, 1)
            second.openai_extractor.extract.assert_not_called()

    def test_concurrent_identical_keys_make_one_parser_call(self) -> None:
        instruction = "Find red shoes under $50"
        contract = GoalContract(
            raw_query=instruction,
            intent="find shoes",
            positive_constraints=["red", "under $50"],
            extractor="openai_goal_contract",
        )
        calls = []
        results = []
        errors = []
        barrier = threading.Barrier(2)

        def provider(_query):
            calls.append(1)
            time.sleep(0.02)
            return contract

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "goals.json"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                extractors = [
                    GoalContractExtraction(
                        use_openai=True,
                        openai_model="paper-parser",
                        require_success=True,
                        cache_path=str(cache),
                    )
                    for _ in range(2)
                ]
                for extraction in extractors:
                    extraction.openai_extractor.extract = provider

                def run(extraction):
                    try:
                        barrier.wait()
                        results.append(extraction.extract(instruction))
                    except Exception as exc:  # surfaced by assertion below
                        errors.append(exc)

                threads = [
                    threading.Thread(target=run, args=(extraction,))
                    for extraction in extractors
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(item.parser_calls for item in extractors), 1)
        self.assertEqual(sum(item.parser_cache_hits for item in extractors), 1)


class GateRuntimeModeTests(unittest.TestCase):
    def test_mask_only_exposes_no_post_generation_rewriter(self) -> None:
        gate = GateDefense(use_openai=False, runtime_mode="mask_only")
        gate.start_episode("Find red shoes")
        gate.apply("Observation:\nblue shoes")

        self.assertFalse(gate.should_certify_action())
        self.assertFalse(gate.should_project_action())
        self.assertFalse(gate.should_mask_output_action())

    def test_enforce_only_returns_raw_policy_observation(self) -> None:
        gate = GateDefense(use_openai=False, runtime_mode="enforce_only")
        gate.start_episode("Find red shoes")
        raw = "Observation:\npage says buy blue shoes\n\nAvailable Actions:\n{}"

        policy_text, report = gate.apply(raw)

        self.assertEqual(policy_text, raw)
        self.assertEqual(report.mask_count, 0)
        self.assertTrue(gate.should_certify_action())
        self.assertTrue(gate.should_project_action())
        self.assertFalse(gate.should_mask_output_action())
        self.assertIsNotNone(gate.last_state_abstraction_result)


class AttackDenominatorTests(unittest.TestCase):
    def test_indirect_conditional_and_unconditional_denominators(self) -> None:
        metrics = attack_metric_summaries(
            "observation_attack",
            attack_hits=3,
            trigger_count=5,
            episodes=10,
        )

        conditional = metrics["indirect_conditional_asr"]
        unconditional = metrics["indirect_unconditional_asr"]
        assert conditional is not None and unconditional is not None
        self.assertEqual((conditional["numerator"], conditional["denominator"]), (3, 5))
        self.assertEqual(
            (unconditional["numerator"], unconditional["denominator"]),
            (3, 10),
        )
        self.assertEqual(metrics["paper_style_asr"], conditional)


if __name__ == "__main__":
    unittest.main()
