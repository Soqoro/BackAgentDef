from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'scripts/seek'))
from prepare_alternative_observation_queries import select

class AlternativeTests(unittest.TestCase):
    def group(self, query, category='sport'):
        return dict(category=category, product_category=query, query=query, group_hash=query,
                    target_products=1, other_products=2)
    def test_bounded_deterministic_distinct_and_exclusions(self):
        rows = [self.group(q) for q in ['Z','B','A','C','D','mens cologne','T-shirts']]
        rows.append(self.group('A'))
        selected, excluded = select(rows)
        self.assertEqual([g['query'] for g in selected], ['A','B','C'])
        self.assertEqual(select(list(reversed(rows))), (selected, excluded))
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(g['task_instruction'] is None and g['runtime_price_limit'] is None
                            and g['approved_slot'] is False for g in selected))
    def test_empty_is_not_filled_with_old_task(self):
        self.assertEqual(select([self.group('eau de toilette')])[0], [])

if __name__ == '__main__':unittest.main()
