from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import diagnose_lexical_v2_wording as m

class WordingTests(unittest.TestCase):
    def test_natural_wording_has_no_approval(self):
        text='Find a black bag under 40 dollars.'
        r=m.inspect(text,'black');self.assertEqual(len(r['required_mentions']),1)
        self.assertEqual(r['explicit_fields'],[]);self.assertFalse(r['eligible'])
        a,b=r['required_mentions'][0]['span'];self.assertEqual(text[a:b],'black')
    def test_fields_and_case(self):
        r=m.inspect('Color: BLACK, colour: blue.','black')
        self.assertEqual([f['text'] for f in r['explicit_fields']],['BLACK','blue'])
        self.assertTrue(r['multiple_distinct_colors'])
    def test_no_substrings(self):
        self.assertEqual(m.inspect('blackberry juice and infrared light','black')['color_mentions'],[])
    def test_conflicts_flags_not_decisions(self):
        r=m.inspect('Not red; either black or white.','black')
        self.assertTrue(r['negation_cues']);self.assertTrue(r['alternative_cues']);self.assertTrue(r['multiple_distinct_colors'])
        self.assertFalse(r['positive_requirement_verified'])
    def test_repeated_and_compound(self):
        r=m.inspect('black-and-white with black trim','black')
        self.assertTrue(r['repeated_color_mentions']);self.assertEqual(len(r['color_mentions']),3)
    def test_missing_instruction_count(self):
        result=m.diagnose({'decisions':[{'id':'missing'}]},{'required':'black'})
        self.assertEqual(result['summary']['missing_instructions'],1)
    def test_duplicate_decisions_rejected(self):
        with self.assertRaises(m.Invalid):m.diagnose({'decisions':[{'id':'a'},{'id':'a'}]},{'required':'black'})

if __name__=='__main__':unittest.main()
