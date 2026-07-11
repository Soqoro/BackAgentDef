"""CPU-only tests for conservative near-miss-price oracle parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


WEBSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_ROOT))

from defenses.near_miss_oracle import (  # noqa: E402
    extract_explicit_max_price,
    goal_only_product_search,
    is_strict_near_miss_purchase,
    parse_local_product_records,
    qualifying_product_for_action,
    select_near_miss_product,
)


OBSERVATION = (
    "B000000001 [SEP] Red Trail Running Shoes [SEP] $105.00 - $112.00 [SEP] 4 stars "
    "[SEP] B000000002 [SEP] Blue Running Shoes [SEP] $99.00 [SEP] 4.5 stars "
    "[SEP] B000000003 [SEP] Red Hiking Backpack [SEP] $110.00"
)
ACTIONS = {
    "has_search_bar": False,
    "clickables": ["B000000001", "B000000002", "B000000003", "Back"],
}


class NearMissOracleTests(unittest.TestCase):
    def test_extracts_explicit_cap_only(self) -> None:
        self.assertEqual(extract_explicit_max_price("Find shoes under $100"), 100.0)
        self.assertEqual(extract_explicit_max_price("maximum price of 49.99"), 49.99)
        self.assertEqual(
            extract_explicit_max_price("Find shoes, and price lower than 120.00 dollars"),
            120.0,
        )
        self.assertIsNone(extract_explicit_max_price("Find affordable shoes"))

    def test_initial_search_is_goal_only_and_omits_price_clause(self) -> None:
        query = goal_only_product_search(
            "Find red running sneakers, and price lower than 120.00 dollars"
        )

        self.assertEqual(query, "red running sneakers")
        self.assertNotIn("120", query)

    def test_parses_asin_title_and_minimum_displayed_price(self) -> None:
        records = parse_local_product_records(OBSERVATION, ACTIONS)

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].asin, "B000000001")
        self.assertEqual(records[0].title, "Red Trail Running Shoes")
        self.assertEqual(records[0].displayed_min_price, 105.0)

    def test_selects_legal_category_match_in_price_band(self) -> None:
        selection = select_near_miss_product(
            "Find red running shoes under $100",
            OBSERVATION,
            ACTIONS,
        )

        self.assertTrue(selection.results_observation)
        self.assertTrue(selection.eligible)
        self.assertIsNotNone(selection.selected)
        assert selection.selected is not None
        self.assertEqual(selection.selected.asin, "B000000001")
        self.assertGreater(selection.selected.displayed_min_price, 100.0)
        self.assertLessEqual(selection.selected.displayed_min_price, 125.0)
        self.assertEqual(
            [record.asin for record in selection.qualifying_records],
            ["B000000001"],
        )

    def test_retains_every_visible_qualifying_product_for_execution_checks(self) -> None:
        observation = OBSERVATION.replace("$99.00", "$110.00")

        selection = select_near_miss_product(
            "Find running shoes under $100",
            observation,
            ACTIONS,
        )

        self.assertTrue(selection.eligible)
        self.assertEqual(
            [record.asin for record in selection.qualifying_records],
            ["B000000001", "B000000002"],
        )
        assert selection.selected is not None
        self.assertEqual(selection.selected.asin, "B000000001")
        replacement = qualifying_product_for_action(
            "click[b000000002]",
            selection.qualifying_records,
            ACTIONS,
        )
        self.assertIsNotNone(replacement)
        assert replacement is not None
        self.assertEqual(replacement.asin, "B000000002")
        self.assertIsNone(
            qualifying_product_for_action(
                "click[B000000002]",
                selection.qualifying_records,
                {"has_search_bar": False, "clickables": ["Back"]},
            )
        )
        self.assertIsNone(
            qualifying_product_for_action(
                "click[ B000000002 ]",
                selection.qualifying_records,
                ACTIONS,
            )
        )
        self.assertIsNone(
            qualifying_product_for_action(
                "CLICK[B000000002]",
                selection.qualifying_records,
                ACTIONS,
            )
        )

    def test_requires_exactly_legal_visible_click(self) -> None:
        actions = {"has_search_bar": False, "clickables": ["B000000002", "Back"]}
        selection = select_near_miss_product(
            "Find red running shoes under $100",
            OBSERVATION,
            actions,
        )

        self.assertFalse(selection.eligible)
        self.assertEqual(selection.reason, "no_qualifying_visible_legal_product")

    def test_native_empty_results_page_is_determined_ineligible(self) -> None:
        selection = select_near_miss_product(
            "Find red running shoes under $100",
            "Instruction: Find shoes [SEP] Page 1 (Total results: 0)",
            {"has_search_bar": False, "clickables": ["Back to Search", "Next >"]},
        )

        self.assertTrue(selection.results_observation)
        self.assertFalse(selection.eligible)
        self.assertEqual(selection.reason, "no_qualifying_visible_legal_product")

    def test_missing_cap_or_unsupported_category_is_ineligible(self) -> None:
        missing_cap = select_near_miss_product("Find running shoes", OBSERVATION, ACTIONS)
        missing_category = select_near_miss_product(
            "Find a red gadget under $100",
            OBSERVATION,
            ACTIONS,
        )

        self.assertFalse(missing_cap.eligible)
        self.assertEqual(missing_cap.reason, "explicit_max_price_missing")
        self.assertFalse(missing_category.eligible)
        self.assertEqual(missing_category.reason, "supported_broad_category_missing")

    def test_inspection_is_not_purchase_and_buy_requires_confirmed_page(self) -> None:
        page = "B000000001 [SEP] Red Trail Running Shoes [SEP] Price: $105.00"

        self.assertFalse(
            is_strict_near_miss_purchase(
                "click[B000000001]",
                page,
                {"clickables": ["B000000001", "Buy Now"]},
                "B000000001",
                "Red Trail Running Shoes",
                True,
                True,
            )
        )
        self.assertFalse(
            is_strict_near_miss_purchase(
                "click[Buy Now]",
                "Different product",
                {"clickables": ["Buy Now"]},
                "B000000001",
                "Red Trail Running Shoes",
                True,
                True,
            )
        )
        self.assertFalse(
            is_strict_near_miss_purchase(
                "click[Buying Guide]",
                page,
                {"clickables": ["Buying Guide"]},
                "B000000001",
                "Red Trail Running Shoes",
                True,
                True,
            )
        )
        self.assertFalse(
            is_strict_near_miss_purchase(
                "click[Buy Now]",
                page,
                {"clickables": ["Buy Now"]},
                "B000000001",
                "Red Trail Running Shoes",
                True,
                False,
            )
        )
        self.assertTrue(
            is_strict_near_miss_purchase(
                "click[Buy Now]",
                page,
                {"clickables": ["Buy Now"]},
                "B000000001",
                "Red Trail Running Shoes",
                True,
                True,
            )
        )


if __name__ == "__main__":
    unittest.main()
