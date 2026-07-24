import unittest

from choice_integrity.public_ledger import (
    collect_fixed_shortlist,
    parse_description_page,
    parse_feature_page,
    parse_item_page,
    parse_search_results,
    parse_text_item_page,
    parse_text_search_results,
    parse_text_subpage,
)
from choice_integrity.webshop_adapter import hard_feasibility
from choice_integrity.webshop_adapter import parse_price


SEARCH_HTML = """
<div class="list-group-item">
  <h4 class="product-asin"><a class="product-link">B000000001</a></h4>
  <h4 class="product-title">Black Sneaker</h4>
  <h5 class="product-brand">Brand: Acme</h5>
  <h5 class="product-price">$10.00</h5>
  <h5 class="product-rating">Rating: 4.5</h5>
  <h5 class="product-availability">Availability: In Stock</h5>
</div>
"""

ITEM_HTML = """
<h4 class="product-asin">ASIN: B000000001</h4>
<h2 class="product-title">Black Sneaker</h2>
<h4 class="product-brand">Brand: Acme</h4>
<h4 class="product-price">Price: $10.00</h4>
<h4 class="product-rating">Rating: 4.5</h4>
<h4 class="product-availability">Availability: In Stock</h4>
<input type="radio" name="size" value="m">
<input type="radio" name="size" value="l">
"""

DESCRIPTION_HTML = """
<p class="product-info">A black fashion sneaker for everyday wear.</p>
"""

FEATURES_HTML = """
<p class="product-info">Lightweight upper</p>
<p class="product-info">Cushioned sole</p>
"""

SEARCH_TEXT = (
    "Instruction: Compare B000000099 [SEP] Back to Search [SEP] "
    "Page 2 (Total results: 20) [SEP] < Prev [SEP] Next > [SEP] "
    "B000000001 [SEP] Black Sneaker [SEP] Brand: Acme [SEP] "
    "$10.00 [SEP] Rating: 4.5 [SEP] Availability: In Stock [SEP] "
    "B000000002 [SEP] Blue Sneaker [SEP] Brand: Other [SEP] "
    "$12.50 [SEP] Rating: 4.1 [SEP] Availability: Only 2 left"
)

ITEM_TEXT = (
    "Instruction: Find a shoe [SEP] Back to Search [SEP] < Prev [SEP] "
    "size [SEP] m [SEP] l [SEP] ASIN: B000000001 [SEP] "
    "Black Sneaker [SEP] Brand: Acme [SEP] Price: $10.00 [SEP] "
    "Rating: 4.5 [SEP] Availability: In Stock [SEP] Description [SEP] "
    "Features [SEP] Reviews [SEP] Buy Now"
)


class FakePublicEnv:
    """Minimal public env surface; deliberately has no ``server`` attribute."""

    def __init__(self):
        self.page = "start"
        self.expected = [
            ("search[black sneaker m]", "results"),
            ("click[B000000001]", "item"),
            ("click[Description]", "description"),
            ("click[< Prev]", "item"),
            ("click[Features]", "features"),
            ("click[< Prev]", "item"),
            ("click[< Prev]", "results"),
        ]

    @property
    def state(self):
        raise AssertionError("the public collector must not inspect env.state")

    def get_available_actions(self):
        clickables = {
            "start": [],
            "results": ["b000000001"],
            "item": ["description", "features", "< prev"],
            "description": ["< prev"],
            "features": ["< prev"],
        }[self.page]
        return {
            "has_search_bar": self.page in {"start", "results"},
            "clickables": clickables,
        }

    def step(self, action):
        expected_action, next_page = self.expected.pop(0)
        if action != expected_action:
            raise AssertionError(f"expected {expected_action}, got {action}")
        self.page = next_page
        observation = {
            "start": "",
            "results": SEARCH_HTML,
            "item": ITEM_HTML,
            "description": DESCRIPTION_HTML,
            "features": FEATURES_HTML,
        }[next_page]
        return observation, 0.0, False, None


class PublicEvidenceParserTests(unittest.TestCase):
    def test_hidden_annotations_cannot_supply_mandatory_attribute_evidence(self):
        feasible, _options, failures = hard_feasibility(
            goal={
                "query": "sneaker",
                "attributes": ["waterproof"],
                "options": {},
                "price_upper": 50,
            },
            product={
                "query": "sneaker",
                "Attributes": [],
                "Title": "plain shoe",
                "Brand": "generic",
                "Description": "waterproof",
                "options": {},
            },
            price=20,
        )

        self.assertFalse(feasible)
        self.assertTrue(any("waterproof" in item for item in failures))

    def test_scalar_price_is_parsed_and_range_is_rejected(self):
        self.assertEqual(parse_price("$1,234.50"), 1234.5)
        self.assertIsNone(parse_price("$10.00 to $20.00"))

    def test_dom_parsers_recover_only_rendered_fields(self):
        result = parse_search_results(SEARCH_HTML)[0]
        item = parse_item_page(ITEM_HTML)
        description = parse_description_page(DESCRIPTION_HTML)
        features = parse_feature_page(FEATURES_HTML)

        self.assertEqual(result["asin"], "B000000001")
        self.assertEqual(result["price"], 10.0)
        self.assertEqual(result["rating"], 4.5)
        self.assertEqual(item["options"], {"size": ["m", "l"]})
        self.assertIn("black fashion sneaker", description.lower())
        self.assertEqual(features, ["Lightweight upper", "Cushioned sole"])

    def test_text_parsers_use_clickable_identity_and_public_labels(self):
        results = parse_text_search_results(
            SEARCH_TEXT,
            allowed_asins=("B000000001", "B000000002"),
        )
        item = parse_text_item_page(ITEM_TEXT)
        description = parse_text_subpage(
            "Instruction [SEP] Back to Search [SEP] < Prev [SEP] "
            "A black fashion sneaker for everyday wear."
        )
        features = parse_text_subpage(
            "Instruction [SEP] Back to Search [SEP] < Prev [SEP] "
            "Lightweight upper [SEP] Cushioned sole"
        )

        self.assertEqual(
            [record["asin"] for record in results],
            ["B000000001", "B000000002"],
        )
        self.assertEqual(results[0]["page"], 2)
        self.assertEqual(results[0]["result_rank"], 1)
        self.assertEqual(results[0]["price"], 10.0)
        self.assertEqual(results[1]["rating"], 4.1)
        self.assertEqual(item["asin"], "B000000001")
        self.assertEqual(item["brand"], "Acme")
        self.assertEqual(item["price"], 10.0)
        self.assertEqual(
            description,
            ["A black fashion sneaker for everyday wear."],
        )
        self.assertEqual(features, ["Lightweight upper", "Cushioned sole"])

    def test_text_search_parser_rejects_non_clickable_asin_shaped_text(self):
        records = parse_text_search_results(
            "B000000099 [SEP] malicious instruction fragment [SEP] "
            "B000000001 [SEP] Visible Product [SEP] $8.00",
            allowed_asins=("B000000001",),
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["asin"], "B000000001")

    def test_collector_needs_no_catalog_or_server_access_and_counts_actions(self):
        env = FakePublicEnv()
        candidates, trace = collect_fixed_shortlist(
            env,
            hard_constraints={
                "query": "sneaker",
                "attributes": ["black"],
                "options": {"size": "m"},
                "price_upper": 50,
            },
            canonical_query="black sneaker m",
            shortlist_size=1,
            page_size=10,
        )

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].feasible)
        self.assertEqual(dict(candidates[0].options), {"size": "m"})
        self.assertEqual(len(trace), 7)
        self.assertNotIn(
            "click[Attributes]",
            [item["action"] for item in trace],
        )
        self.assertTrue(
            all(
                item["observation_sha256"] == item["html_sha256"]
                for item in trace
            )
        )
        self.assertEqual(env.expected, [])


if __name__ == "__main__":
    unittest.main()
