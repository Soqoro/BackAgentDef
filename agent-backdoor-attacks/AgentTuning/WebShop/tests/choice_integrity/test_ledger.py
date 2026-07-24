import unittest

from choice_integrity.ledger import (
    CandidateLedger,
    LedgerError,
    LedgerIncompleteError,
)
from choice_integrity.schema import Candidate, Preference


def candidate(
    asin,
    *,
    options=None,
    feasible=True,
    price=None,
    rating=None,
    evidence=None,
    metadata=None,
):
    return Candidate(
        asin=asin,
        options=options or {},
        feasible=feasible,
        price=price,
        rating=rating,
        evidence=evidence or {},
        metadata=metadata or {},
    )


class CandidateLedgerTests(unittest.TestCase):
    def test_deduplicates_product_option_identity_and_keeps_provenance(self):
        first = candidate(
            "A",
            options={"size": "M", "color": "blue"},
            price=12,
            evidence={"price": {"text": "$12"}},
        )
        reordered = candidate(
            "A",
            options={"color": "blue", "size": "M"},
            price=12,
            evidence={"price": {"text": "$12.00"}},
        )

        ledger = CandidateLedger.from_candidates(
            (first, reordered),
            source="results_page",
        )

        self.assertEqual(len(ledger.entries()), 1)
        entry = ledger.entries()[0]
        self.assertEqual(entry.candidate_id, first.candidate_id)
        self.assertEqual(entry.sources, {"results_page"})
        self.assertEqual(len(entry.observations), 2)
        self.assertEqual(dict(entry.options), {"color": "blue", "size": "M"})

    def test_evaluator_labels_and_candidate_metadata_never_enter_public_view(self):
        item = candidate(
            "A",
            price=10,
            evidence={
                "price": {
                    "text": "$10",
                    "winner_id": "SECRET-WINNER",
                    "nested": {"attacker_target_ids": ["SECRET-TARGET"]},
                    "target_candidate_id": "SECRET-TARGET",
                },
                "ground_truth": "SECRET",
                "unknown_smuggling_field": "SECRET",
            },
            metadata={
                "winner_ids": ["SECRET-WINNER"],
                "attacker_target_ids": ["SECRET-TARGET"],
            },
        )

        ledger = CandidateLedger.from_candidates(
            (item,),
            source="product_page",
        )
        rendered = repr(ledger.to_dict())
        public = ledger.public_candidates()[0]

        self.assertNotIn("SECRET", rendered)
        self.assertEqual(dict(public.metadata), {})
        self.assertNotIn("winner", repr(public.to_dict()).lower())
        self.assertNotIn("attacker", repr(public.to_dict()).lower())

    def test_evaluator_label_shaped_sources_are_rejected(self):
        item = candidate("A", price=10)
        for source in ("winner_table", "attacker-target-oracle", "ground truth"):
            with self.subTest(source=source):
                with self.assertRaises(LedgerError):
                    CandidateLedger.from_candidates((item,), source=source)

    def test_requires_fixed_shortlist_coverage_and_comparison_fields(self):
        complete_item = candidate("A", price=10)
        ledger = CandidateLedger.from_candidates(
            (complete_item,),
            source="results_page",
            comparison_complete=False,
        )
        with self.assertRaisesRegex(
            LedgerIncompleteError, "shortlist is incomplete"
        ):
            ledger.winners(Preference.price_min())

        ledger.mark_comparison_complete()
        self.assertEqual(
            tuple(entry.candidate_id for entry in ledger.winners(
                Preference.price_min()
            )),
            (complete_item.candidate_id,),
        )

        missing_rating = CandidateLedger.from_candidates(
            (candidate("B", price=9, rating=None),),
            source="results_page",
        )
        with self.assertRaises(LedgerIncompleteError) as context:
            missing_rating.winners(Preference.rating_max_price_tiebreak())
        self.assertIn("rating", repr(context.exception.missing))

    def test_conflicting_public_observations_fail_closed(self):
        ledger = CandidateLedger.from_candidates(
            (candidate("A", price=10),),
            source="results_page",
        )
        ledger.upsert(
            candidate("A", price=11),
            source="product_page",
        )

        self.assertIn("price", ledger.entries()[0].ambiguous_fields)
        with self.assertRaises(LedgerIncompleteError):
            ledger.winners(Preference.price_min())

    def test_conflicting_feasibility_cannot_be_treated_as_feasible(self):
        ledger = CandidateLedger.from_candidates(
            (candidate("A", feasible=True, price=10),),
            source="results_page",
        )
        ledger.upsert(
            candidate("A", feasible=False, price=10),
            source="product_page",
        )

        self.assertIn("feasible", ledger.entries()[0].ambiguous_fields)
        with self.assertRaises(LedgerIncompleteError):
            ledger.winners(Preference.price_min())

    def test_runtime_public_observation_updates_only_an_existing_identity(self):
        item = candidate("B000000001", price=None)
        ledger = CandidateLedger.from_candidates(
            (item,),
            source="comparison_session",
        )

        entry = ledger.observe_existing(
            item.candidate_id,
            source="policy_public_item_page",
            fields={"price": 10, "brand": "Acme"},
            observed_fields=("availability",),
            evidence={
                "price": {
                    "value": 10,
                    "winner_id": "SECRET-WINNER",
                },
                "availability": {
                    "value": "In Stock",
                    "target_candidate_id": "SECRET-TARGET",
                },
            },
        )

        self.assertEqual(len(ledger.entries()), 1)
        self.assertEqual(entry.price, 10.0)
        self.assertEqual(entry.brand, "Acme")
        self.assertIn("policy_public_item_page", entry.sources)
        self.assertNotIn("SECRET", repr(ledger.to_dict()))
        self.assertEqual(
            ledger.entries_for_asin("b000000001"),
            (entry,),
        )

        with self.assertRaises(LedgerIncompleteError):
            ledger.observe_existing(
                '{"asin":"B000000099","options":{}}',
                source="policy_public_item_page",
                fields={"price": 1},
            )
        self.assertEqual(len(ledger.entries()), 1)

    def test_runtime_public_conflict_is_ambiguous_and_identity_is_immutable(self):
        item = candidate("B000000001", price=10)
        ledger = CandidateLedger.from_candidates(
            (item,),
            source="comparison_session",
        )

        ledger.observe_existing(
            item,
            source="policy_public_search_results",
            fields={"price": 11},
        )

        self.assertIn("price", ledger.require(item).ambiguous_fields)
        with self.assertRaises(LedgerError):
            ledger.observe_existing(
                item,
                source="policy_public_item_page",
                fields={"title": "Rendered title"},
                observed_fields=("feasible",),
            )
        self.assertTrue(ledger.require(item).feasible)
        self.assertIsNone(ledger.require(item).title)

    def test_hard_infeasible_candidate_never_wins_despite_lower_price(self):
        feasible = candidate("FEASIBLE", feasible=True, price=20)
        impossible = candidate("IMPOSSIBLE", feasible=False, price=1)
        ledger = CandidateLedger.from_candidates(
            (impossible, feasible),
            source="results_page",
        )

        winners = ledger.winners(Preference.price_min())

        self.assertEqual(
            tuple(entry.candidate_id for entry in winners),
            (feasible.candidate_id,),
        )
        self.assertEqual(
            ledger.dominators(feasible, Preference.price_min()),
            (),
        )

    def test_rejects_non_candidate_ingestion(self):
        ledger = CandidateLedger()
        with self.assertRaisesRegex(TypeError, "Candidate"):
            ledger.upsert(
                {"asin": "A", "winner": True},
                source="results_page",
            )


if __name__ == "__main__":
    unittest.main()
