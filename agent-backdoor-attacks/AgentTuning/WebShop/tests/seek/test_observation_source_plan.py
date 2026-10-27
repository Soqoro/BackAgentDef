import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import prepare_observation_source_plan as m

class PlanTests(unittest.TestCase):
    def config(self):return m.read_json(m.CONFIG)
    def task(self):return dict(id='mens_edt',price_upper=50,instruction='fixed task')
    def test_schedule_is_fixed_and_bounded(self):
        config=self.config();before=copy.deepcopy(config);rows=m.schedule(config,self.task())
        self.assertEqual(config,before)
        self.assertEqual(len(rows),3)
        self.assertEqual([r['query'] for r in rows],m.QUERIES)
        self.assertEqual(sum(r['maximum_pages'] for r in rows),6)
    def test_expansion_rejected(self):
        config=self.config();config['maximum_pages_per_query']=3
        with self.assertRaises(m.Invalid):m.schedule(config,self.task())
    def test_target_query_substitution_rejected(self):
        config=self.config();config['queries'][0]='adidas fragrance'
        with self.assertRaises(m.Invalid):m.schedule(config,self.task())
    def test_execution_not_approved_by_preparation(self):
        config=self.config();config['execution_authorized']=True
        with self.assertRaises(m.Invalid):m.schedule(config,self.task())
if __name__=='__main__':unittest.main()
