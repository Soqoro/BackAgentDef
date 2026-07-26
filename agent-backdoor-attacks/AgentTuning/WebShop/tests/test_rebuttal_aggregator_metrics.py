import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


WEBSHOP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBSHOP_DIR))

import aggregate_rebuttal as aggregate  # noqa: E402


class RebuttalAggregatorMetricTests(unittest.TestCase):
    def test_wilson_interval_known_values_and_invalid_counts(self):
        low, high = aggregate.wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.236593, places=6)
        self.assertAlmostEqual(high, 0.763407, places=6)

        zero_low, zero_high = aggregate.wilson_interval(0, 10)
        self.assertAlmostEqual(zero_low, 0.0, places=12)
        self.assertAlmostEqual(zero_high, 0.277533, places=6)
        self.assertIsNone(aggregate.wilson_interval(0, 0))

        with self.assertRaises(ValueError):
            aggregate.wilson_interval(11, 10)

    def test_paired_bootstrap_is_seeded_and_preserves_pairing(self):
        candidate = [1.0, 2.0, 4.0, 2.0]
        baseline = [0.0, 1.0, 1.0, 2.0]
        first = aggregate.paired_bootstrap_aer_difference(
            candidate, baseline, samples=500
        )
        second = aggregate.paired_bootstrap_aer_difference(
            candidate, baseline, samples=500
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 1.25)
        self.assertLessEqual(first[1], first[0])
        self.assertLessEqual(first[0], first[2])

        self.assertEqual(
            aggregate.paired_bootstrap_aer_difference(
                [2.0, 3.0], [1.0, 2.0], samples=20
            ),
            (1.0, 1.0, 1.0),
        )

    def test_clean_change_counts_are_paired_by_task_id(self):
        candidate = {"10": 0.8, "11": 0.5, "12": 0.1}
        baseline = {"12": 0.2, "10": 0.4, "11": 0.5}
        self.assertEqual(
            aggregate.clean_change_counts(candidate, baseline), (1, 1, 1)
        )

        with self.assertRaises(ValueError):
            aggregate.clean_change_counts(candidate, {"10": 0.4})

    def test_known_seed_mismatch_is_not_reported_as_paired(self):
        episode = {
            "task_id": 1,
            "reward": 0.5,
            "exact_reward_task_success": False,
        }
        baseline = aggregate.RunRecord.from_raw(
            Path("none.json"),
            {
                "method": "none",
                "attack_type": "clean",
                "seed": 7,
                "task_ids": [1],
                "per_episode": [episode],
            },
        )
        candidate = aggregate.RunRecord.from_raw(
            Path("gate.json"),
            {
                "method": "gate/full",
                "attack_type": "clean",
                "seed": 42,
                "task_ids": [1],
                "per_episode": [episode],
            },
        )
        rows = aggregate.aggregate_runs([baseline, candidate], bootstrap_samples=10)
        self.assertIsNone(rows[1]["aer_diff_vs_none"])
        self.assertIsNone(rows[1]["paired_episode_count"])

    def test_inconsistent_summary_and_episode_data_are_rejected(self):
        with self.assertRaisesRegex(aggregate.AggregationError, "episode_count"):
            aggregate.RunRecord.from_raw(
                Path("bad_count.json"),
                {
                    "method": "none",
                    "attack_type": "clean",
                    "episode_count": 2,
                    "task_ids": [1],
                    "per_episode": [{"task_id": 1, "reward": 0.5}],
                },
            )

        bad_aer = aggregate.RunRecord.from_raw(
            Path("bad_aer.json"),
            {
                "method": "none",
                "attack_type": "clean",
                "aer": 1.0,
                "task_ids": [1],
                "per_episode": [{"task_id": 1, "reward": 0.5}],
            },
        )
        with self.assertRaisesRegex(aggregate.AggregationError, "reported AER"):
            _ = bad_aer.aer

        bad_metric = aggregate.RunRecord.from_raw(
            Path("bad_metric.json"),
            {
                "method": "none",
                "attack_type": "clean",
                "task_ids": [1],
                "exact_reward_task_success_rate": {
                    "numerator": 1,
                    "denominator": 1,
                    "rate": 1.0,
                },
                "per_episode": [
                    {
                        "task_id": 1,
                        "reward": 0.5,
                        "exact_reward_task_success": False,
                    }
                ],
            },
        )
        with self.assertRaisesRegex(aggregate.AggregationError, "reported task_success"):
            bad_metric.metric("task_success")

        contradictory_rate = aggregate.RunRecord.from_raw(
            Path("bad_rate.json"),
            {
                "method": "none",
                "attack_type": "clean",
                "exact_reward_task_success_rate": {
                    "numerator": 1,
                    "denominator": 2,
                    "rate": 0.9,
                    "percent": 90.0,
                },
            },
        )
        with self.assertRaisesRegex(aggregate.AggregationError, "disagrees with counts"):
            contradictory_rate.metric("task_success")

    def test_debug_step_booleans_reproduce_action_metrics(self):
        run = aggregate.RunRecord.from_raw(
            Path("debug.jsonl"),
            {
                "method": "none",
                "attack_type": "query_attack",
                "task_ids": [1],
                "per_episode": [
                    {
                        "task_id": 1,
                        "reward": 0.0,
                        "steps": [
                            {
                                "request_error": None,
                                "valid_action": True,
                                "unparsable_action": False,
                                "proposal_is_malicious": True,
                                "executed_is_malicious": False,
                                "step_intervened": True,
                                "repair_attempted": True,
                                "repair_succeeded": True,
                                "judge_called": False,
                                "judge_failed": False,
                                "judge_replaced": False,
                                "added_runtime_seconds": 0.1,
                            },
                            {
                                "request_error": "network failure",
                                "valid_action": False,
                                "unparsable_action": False,
                                "proposal_is_malicious": False,
                                "executed_is_malicious": False,
                                "step_intervened": False,
                                "repair_attempted": False,
                                "repair_succeeded": False,
                                "judge_called": False,
                                "judge_failed": False,
                                "judge_replaced": False,
                                "added_runtime_seconds": 0.2,
                            },
                        ],
                    }
                ],
            },
        )
        self.assertEqual(
            (run.metric("valid_action").numerator, run.metric("valid_action").denominator),
            (1, 1),
        )
        row = aggregate.aggregate_runs([run], bootstrap_samples=10)[0]
        self.assertEqual(row["repair_call_count"], 1)
        self.assertEqual(row["repair_success_count"], 1)
        self.assertEqual(row["judge_call_count"], 0)
        self.assertAlmostEqual(row["mean_added_runtime_seconds"], 0.15)
        self.assertAlmostEqual(row["median_added_runtime_seconds"], 0.15)
        self.assertAlmostEqual(row["p95_added_runtime_seconds"], 0.195)
        self.assertEqual(
            (
                run.metric("proposed_attack").numerator,
                run.metric("proposed_attack").denominator,
            ),
            (1, 1),
        )
        self.assertEqual(
            (
                run.metric("step_intervention").numerator,
                run.metric("step_intervention").denominator,
            ),
            (1, 1),
        )

    def test_explicit_null_metadata_is_not_overridden_by_cli_defaults(self):
        run = aggregate.RunRecord.from_raw(
            Path("actual.summary.json"),
            {
                "method": "legal_repair",
                "runtime_mode": None,
                "gate_runtime_mode": None,
                "parser_model_requested": None,
                "parser_model_actual": None,
                "judge_model": None,
                "attack_type": "clean",
                "task_ids": [1],
                "per_episode": [{"task_id": 1, "reward": 0.0}],
                "cli_arguments": {
                    "gate_runtime_mode": "full",
                    "gate_openai_model": "should-not-leak",
                    "judge_model": "should-not-leak",
                },
            },
        )
        self.assertIsNone(run.runtime_mode)
        row = aggregate.aggregate_runs([run], bootstrap_samples=10)[0]
        self.assertIsNone(row["parser_model_requested"])
        self.assertIsNone(row["parser_model_actual"])
        self.assertIsNone(row["judge_model"])

    def test_legacy_gate_ablation_is_not_relabelled_as_runtime_mode(self):
        run = aggregate.RunRecord.from_raw(
            Path("legacy-gate.json"),
            {
                "method": "gate",
                "gate_ablation": "no_m2",
                "attack_type": "clean",
                "task_ids": [1],
                "per_episode": [{"task_id": 1, "reward": 0.0}],
            },
        )

        self.assertIsNone(run.runtime_mode)

    def test_non_gate_debug_metadata_cannot_invent_a_gate_runtime_mode(self):
        run = aggregate.RunRecord.from_raw(
            Path("none.debug.jsonl"),
            {
                "method": "none",
                "gate_runtime_mode": "full",
                "attack_type": "clean",
                "task_ids": [1],
                "per_episode": [{"task_id": 1, "reward": 0.0}],
            },
        )

        self.assertIsNone(run.runtime_mode)

    def test_mismatched_task_ids_refuse_aggregation_without_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "none.json"
            second_path = root / "gate.json"
            first_path.write_text(
                json.dumps(
                    {
                        "method": "none",
                        "attack_type": "query_attack",
                        "task_ids": [1, 2],
                        "episode_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    {
                        "method": "gate",
                        "attack_type": "direct",
                        "task_ids": [1, 3],
                        "episode_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            runs = aggregate.load_runs([first_path, second_path])

            with self.assertRaisesRegex(
                aggregate.AggregationError, "mismatched task IDs"
            ):
                aggregate.validate_identical_task_ids(runs)
            warnings = aggregate.validate_identical_task_ids(
                runs, allow_mismatched=True
            )
            self.assertTrue(warnings)

    def test_jsonl_separates_conditional_and_unconditional_indirect_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "indirect.jsonl"
            records = [
                {
                    "task_id": 7,
                    "method": "none",
                    "attack_type": "observation_attack",
                    "final_reward": 0.4,
                    "paper_style_attack_hit": True,
                    "observation_trigger_seen": True,
                    "exact_reward_task_success": False,
                },
                {
                    "task_id": 8,
                    "method": "none",
                    "attack_type": "observation_attack",
                    "final_reward": 1.0,
                    "paper_style_attack_hit": False,
                    "observation_trigger_seen": False,
                    "exact_reward_task_success": True,
                },
                {
                    "task_id": 9,
                    "method": "none",
                    "attack_type": "observation_attack",
                    "final_reward": 0.0,
                    "paper_style_attack_hit": False,
                    "observation_trigger_seen": True,
                    "exact_reward_task_success": False,
                },
            ]
            log_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            run = aggregate.load_runs([log_path])[0]
            conditional = run.metric("indirect_conditional_asr")
            unconditional = run.metric("indirect_unconditional_asr")
            trigger = run.metric("trigger_exposure")
            self.assertEqual(
                (conditional.numerator, conditional.denominator), (1, 2)
            )
            self.assertEqual(
                (unconditional.numerator, unconditional.denominator), (1, 3)
            )
            self.assertEqual((trigger.numerator, trigger.denominator), (2, 3))

    def test_near_miss_oracle_metrics_use_eligible_denominator_and_render(self):
        run = aggregate.RunRecord.from_raw(
            Path("near_miss.summary.json"),
            {
                "method": "gate/full",
                "runtime_mode": "full",
                "attack_type": "observation_attack",
                "stress_test": "action_oracle",
                "oracle_mode": "indirect_oracle",
                "oracle_strategy": "near_miss_price",
                "seed": 42,
                "task_ids": [10, 11, 12, 13],
                "episode_count": 4,
                "oracle_eligible_count": 2,
                "oracle_eligibility_determined_count": 3,
                "oracle_eligible_rate": {
                    "numerator": 2,
                    "denominator": 3,
                    "rate": 2 / 3,
                    "percent": 200 / 3,
                },
                "oracle_violating_product_click_rate": {
                    "numerator": 1,
                    "denominator": 2,
                    "rate": 0.5,
                    "percent": 50.0,
                },
                "oracle_strict_purchase_rate": {
                    "numerator": 1,
                    "denominator": 2,
                    "rate": 0.5,
                    "percent": 50.0,
                },
                "per_episode": [
                    {
                        "task_id": 10,
                        "reward": 0.0,
                        "oracle_eligible": True,
                        "oracle_violating_product_click": True,
                        "oracle_strict_purchase": False,
                    },
                    {
                        "task_id": 11,
                        "reward": 0.0,
                        "oracle_eligible": False,
                        "oracle_violating_product_click": False,
                        "oracle_strict_purchase": False,
                    },
                    {
                        "task_id": 12,
                        "reward": 0.0,
                        "oracle_eligible": True,
                        "oracle_violating_product_click": False,
                        "oracle_strict_purchase": True,
                    },
                    {
                        "task_id": 13,
                        "reward": 0.0,
                        "oracle_eligible": None,
                        "oracle_violating_product_click": False,
                        "oracle_strict_purchase": False,
                    },
                ],
            },
        )

        eligible = run.metric("oracle_eligible")
        violating = run.metric("oracle_violating_click")
        purchase = run.metric("oracle_strict_purchase")
        self.assertEqual((eligible.numerator, eligible.denominator), (2, 3))
        self.assertEqual((violating.numerator, violating.denominator), (1, 2))
        self.assertEqual((purchase.numerator, purchase.denominator), (1, 2))
        self.assertIsNotNone(eligible.interval)
        self.assertIsNotNone(violating.interval)

        rows = aggregate.aggregate_runs([run], bootstrap_samples=10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "near_miss.csv"
            markdown_path = root / "near_miss.md"
            latex_path = root / "near_miss.tex"
            aggregate.write_csv(csv_path, rows)
            aggregate.write_markdown(markdown_path, rows)
            aggregate.write_latex(latex_path, rows)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["oracle_eligible_numerator"], "2")
            self.assertEqual(row["oracle_eligible_denominator"], "3")
            self.assertEqual(row["oracle_violating_click_numerator"], "1")
            self.assertEqual(row["oracle_violating_click_denominator"], "2")
            self.assertNotEqual(row["oracle_strict_purchase_ci_low_percent"], "")
            self.assertIn(
                "Oracle violating click \\| eligible",
                markdown_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Oracle strict purchase | eligible",
                latex_path.read_text(encoding="utf-8"),
            )

    def test_target_brand_oracle_does_not_invent_near_miss_metrics(self):
        run = aggregate.RunRecord.from_raw(
            Path("target_brand.debug.jsonl"),
            {
                "method": "none",
                "attack_type": "query_attack",
                "stress_test": "action_oracle",
                "oracle_strategy": "target_brand",
                "task_ids": [1],
                "per_episode": [
                    {
                        "task_id": 1,
                        "reward": 0.0,
                        "oracle_eligible": None,
                        "oracle_violating_product_click": False,
                        "oracle_strict_purchase": False,
                    }
                ],
            },
        )

        self.assertTrue(run.metric("oracle_eligible").is_missing)
        self.assertTrue(run.metric("oracle_violating_click").is_missing)
        self.assertTrue(run.metric("oracle_strict_purchase").is_missing)

    def test_outputs_raw_counts_pair_clean_rewards_and_leave_missing_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "none.summary.json"
            candidate_path = root / "repair.summary.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "method": "none",
                        "runtime_mode": "none",
                        "attack_type": "clean",
                        "seed": 42,
                        "task_ids": [10, 11, 12],
                        "episode_count": 3,
                        "per_episode": [
                            {
                                "task_id": 10,
                                "reward": 0.1,
                                "exact_reward_task_success": False,
                            },
                            {
                                "task_id": 11,
                                "reward": 0.5,
                                "exact_reward_task_success": False,
                            },
                            {
                                "task_id": 12,
                                "reward": 1.0,
                                "exact_reward_task_success": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    {
                        "method": "legal_repair",
                        "runtime_mode": "none",
                        "attack_type": "clean",
                        "seed": 42,
                        "task_ids": [10, 11, 12],
                        "episode_count": 3,
                        "exact_reward_task_success_rate": {
                            "numerator": 0,
                            "denominator": 3,
                            "rate": 0.0,
                            "percent": 0.0,
                        },
                        "per_episode": [
                            {
                                "task_id": 12,
                                "reward": 0.8,
                                "exact_reward_task_success": False,
                            },
                            {
                                "task_id": 10,
                                "reward": 0.2,
                                "exact_reward_task_success": False,
                            },
                            {
                                "task_id": 11,
                                "reward": 0.5,
                                "exact_reward_task_success": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            runs = aggregate.load_runs([baseline_path, candidate_path])
            aggregate.validate_identical_task_ids(runs)
            rows = aggregate.aggregate_runs(runs, bootstrap_samples=100)
            candidate = rows[1]
            self.assertEqual(candidate["clean_improved"], 1)
            self.assertEqual(candidate["clean_unchanged"], 1)
            self.assertEqual(candidate["clean_harmed"], 1)
            self.assertEqual(candidate["paired_episode_count"], 3)
            self.assertEqual(candidate["task_success"].numerator, 0)
            self.assertEqual(candidate["task_success"].denominator, 3)
            self.assertTrue(candidate["direct_asr"].is_missing)

            csv_path = root / "aggregate.csv"
            markdown_path = root / "aggregate.md"
            latex_path = root / "aggregate.tex"
            aggregate.write_csv(csv_path, rows)
            aggregate.write_markdown(markdown_path, rows)
            aggregate.write_latex(latex_path, rows)

            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[1]["task_success_numerator"], "0")
            self.assertEqual(csv_rows[1]["task_success_denominator"], "3")
            self.assertEqual(csv_rows[1]["direct_asr_numerator"], "")
            self.assertIn(
                "Indirect ASR (cond.)",
                markdown_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "legal\\_repair", latex_path.read_text(encoding="utf-8")
            )

    def test_llm_efficiency_fields_round_trip_without_inventing_cost(self):
        efficiency_fields = {
            "parser_request_count": 7,
            "parser_call_count": 5,
            "parser_api_call_count": 5,
            "parser_cache_hit_count": 2,
            "parser_usage_reported_call_count": 4,
            "parser_usage_missing_call_count": 1,
            "parser_input_token_count": 100,
            "parser_cached_input_token_count": 20,
            "parser_output_token_count": 25,
            "parser_reasoning_token_count": 5,
            "parser_total_token_count": 125,
            "parser_estimated_cost_usd": None,
            "judge_request_count": 11,
            "judge_call_count": 9,
            "judge_cache_hit_count": 2,
            "judge_usage_reported_call_count": 8,
            "judge_usage_missing_call_count": 1,
            "judge_input_token_count": 200,
            "judge_cached_input_token_count": 40,
            "judge_output_token_count": 50,
            "judge_reasoning_token_count": 10,
            "judge_total_token_count": 250,
            "judge_estimated_cost_usd": 0.00125,
            "defense_llm_request_count": 18,
            "defense_llm_api_call_count": 14,
            "defense_llm_cache_hit_count": 4,
            "defense_llm_usage_reported_call_count": 12,
            "defense_llm_usage_missing_call_count": 2,
            "defense_llm_input_token_count": 300,
            "defense_llm_cached_input_token_count": 60,
            "defense_llm_output_token_count": 75,
            "defense_llm_reasoning_token_count": 15,
            "defense_llm_total_token_count": 375,
            "defense_llm_estimated_cost_usd": None,
            "defense_llm_requests_per_episode": 18.0,
            "defense_llm_api_calls_per_episode": 14.0,
            "defense_llm_api_calls_per_action_step": 7.0,
            "defense_llm_estimated_cost_usd_per_episode": None,
            "llm_input_usd_per_million": 2.5,
            "llm_cached_input_usd_per_million": 1.25,
            "llm_output_usd_per_million": 10.0,
            "llm_pricing_as_of": "2026-07-27",
            "llm_pricing_source": "configured-test-rates",
            "defense_action_round_count": 20,
            "gate_runtime_round_count": 12,
            "gate_certification_round_count": 3,
        }
        run = aggregate.RunRecord.from_raw(
            Path("judge.summary.json"),
            {
                "method": "llm_judge",
                "attack_type": "clean",
                "task_ids": [1],
                "episode_count": 1,
                "parser_actual_models": ["parser-a", "parser-b"],
                "judge_actual_models": ["judge-a"],
                **efficiency_fields,
                "cli_arguments": {
                    "parser_estimated_cost_usd": 99.0,
                    "defense_llm_estimated_cost_usd": 99.0,
                    "defense_llm_estimated_cost_usd_per_episode": 99.0,
                },
                "per_episode": [{"task_id": 1, "reward": 0.5}],
            },
        )

        row = aggregate.aggregate_runs([run], bootstrap_samples=10)[0]
        for name, expected in efficiency_fields.items():
            self.assertEqual(row[name], expected, name)
        self.assertEqual(row["parser_actual_models"], ["parser-a", "parser-b"])
        self.assertEqual(row["judge_actual_models"], ["judge-a"])
        self.assertIsNone(row["parser_estimated_cost_usd"])
        self.assertIsNone(row["defense_llm_estimated_cost_usd"])
        self.assertIsNone(row["defense_llm_estimated_cost_usd_per_episode"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "efficiency.csv"
            markdown_path = root / "efficiency.md"
            latex_path = root / "efficiency.tex"
            aggregate.write_csv(csv_path, [row])
            aggregate.write_markdown(markdown_path, [row])
            aggregate.write_latex(latex_path, [row])

            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_row = next(csv.DictReader(handle))
            for name, expected in efficiency_fields.items():
                expected_csv = aggregate._csv_value(expected)
                self.assertEqual(csv_row[name], str(expected_csv), name)
            self.assertEqual(
                csv_row["parser_actual_models"], '["parser-a", "parser-b"]'
            )
            self.assertEqual(csv_row["judge_actual_models"], '["judge-a"]')
            self.assertEqual(csv_row["parser_estimated_cost_usd"], "")
            self.assertEqual(csv_row["defense_llm_estimated_cost_usd"], "")
            self.assertEqual(
                csv_row["defense_llm_estimated_cost_usd_per_episode"], ""
            )

            markdown = markdown_path.read_text(encoding="utf-8")
            latex = latex_path.read_text(encoding="utf-8")
            self.assertIn(
                "20/12/3; 18/14/4; 300/60/75/15/375; —",
                markdown,
            )
            self.assertNotIn("375; $0", markdown)
            self.assertIn(
                "20/12/3; 18/14/4; 300/60/75/15/375; --",
                latex,
            )

    def test_legacy_percentage_does_not_invent_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "defense": "none",
                        "eval_type": "query_attack",
                        "episodes": 80,
                        "paper_style_asr": 12.5,
                    }
                ),
                encoding="utf-8",
            )
            metric = aggregate.load_runs([path])[0].metric("direct_asr")
            self.assertAlmostEqual(metric.proportion, 0.125)
            self.assertIsNone(metric.numerator)
            self.assertIsNone(metric.denominator)
            self.assertIsNone(metric.interval)

            numerator_only_path = Path(directory) / "numerator_only.json"
            numerator_only_path.write_text(
                json.dumps(
                    {
                        "defense": "none",
                        "eval_type": "query_attack",
                        "episodes": 80,
                        "asr_hits": 2,
                    }
                ),
                encoding="utf-8",
            )
            numerator_only = aggregate.load_runs([numerator_only_path])[0].metric(
                "direct_asr"
            )
            self.assertEqual(numerator_only.numerator, 2)
            self.assertIsNone(numerator_only.denominator)
            self.assertIsNone(numerator_only.proportion)
            self.assertIsNone(numerator_only.interval)
            self.assertEqual(aggregate._format_metric(numerator_only), "2/?")


if __name__ == "__main__":
    unittest.main()
