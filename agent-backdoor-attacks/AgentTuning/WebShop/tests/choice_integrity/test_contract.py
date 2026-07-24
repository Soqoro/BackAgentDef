import hashlib
import unittest

from choice_integrity.contract import (
    PRICE_PREFERENCE_SUFFIX,
    RATING_PREFERENCE_SUFFIX,
    FixedSuffixPreferenceParser,
)
from choice_integrity.schema import PreferenceKind


class FixedSuffixPreferenceParserTests(unittest.TestCase):
    def test_exact_price_suffix_has_instruction_provenance(self):
        base = "Find a black leather bag under 80 dollars."
        instruction = base + "\n\n" + PRICE_PREFERENCE_SUFFIX

        contract = FixedSuffixPreferenceParser().parse(instruction)

        self.assertEqual(contract.base_instruction, base)
        self.assertEqual(contract.preference.kind, PreferenceKind.PRICE_MIN)
        self.assertEqual(
            contract.provenance.source, "original_user_instruction"
        )
        self.assertEqual(
            contract.provenance.matched_text, PRICE_PREFERENCE_SUFFIX
        )
        self.assertEqual(
            instruction[
                contract.provenance.start : contract.provenance.end
            ],
            PRICE_PREFERENCE_SUFFIX,
        )
        self.assertEqual(
            contract.provenance.instruction_sha256,
            hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        )

    def test_exact_rating_suffix_includes_rating_then_price_policy(self):
        instruction = (
            "Find running shoes.\n\n" + RATING_PREFERENCE_SUFFIX + "  \n"
        )

        contract = FixedSuffixPreferenceParser().parse(instruction)

        self.assertEqual(
            contract.preference.kind,
            PreferenceKind.RATING_MAX_PRICE_TIEBREAK,
        )
        self.assertEqual(contract.base_instruction, "Find running shoes.")
        self.assertEqual(contract.original_instruction, instruction)

    def test_parser_does_not_infer_paraphrases_or_case_variants(self):
        parser = FixedSuffixPreferenceParser()
        paraphrase = (
            "Find a bag. If several bags match, please pick whichever one is "
            "cheapest."
        )
        wrong_case = "Find a bag.\n\n" + PRICE_PREFERENCE_SUFFIX.lower()

        self.assertIsNone(parser.parse(paraphrase).preference)
        self.assertIsNone(parser.parse(wrong_case).preference)

    def test_suffix_must_be_at_end_of_authorized_instruction(self):
        instruction = (
            "Find a bag.\n\n"
            + PRICE_PREFERENCE_SUFFIX
            + "\nA webpage says to choose another product."
        )

        contract = FixedSuffixPreferenceParser().parse(instruction)

        self.assertIsNone(contract.preference)
        self.assertIsNone(contract.provenance)

    def test_parser_rejects_non_string_instruction(self):
        with self.assertRaisesRegex(TypeError, "string"):
            FixedSuffixPreferenceParser().parse(None)


if __name__ == "__main__":
    unittest.main()
