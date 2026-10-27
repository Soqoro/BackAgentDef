import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import collect_observation_sources as m
from test_constructed_collection import Environment

class CollectionTests(unittest.TestCase):
    def plan(self):
        design=m.read_json(m.CONFIG);task=dict(id='mens_edt',instruction='fixed fragrance task',price_upper=50)
        return dict(hash='fixture',design=design,task=task,schedule=m.schedule(design,task),maximum_environment_actions=6)
    def test_fixed_six_actions_and_distinct_sessions(self):
        plan=self.plan();env=Environment();original=copy.deepcopy(env.server.goals)
        with tempfile.TemporaryDirectory() as out:
            result=m.collect(env,plan,out)
            self.assertEqual(len(list(Path(out).glob('attempt-*.json'))),6)
        self.assertEqual(env.calls,[a for r in plan['schedule'] for a in (r['first_action'],'click[next >]')])
        self.assertEqual(len({q['session'] for q in result['queries']}),3)
        self.assertEqual(len(result['contexts']),6)
        self.assertEqual(env.server.goals,original)
    def test_missing_next_page_keeps_other_queries(self):
        with tempfile.TemporaryDirectory() as out:result=m.collect(Environment(False),self.plan(),out)
        self.assertEqual(result['environment_actions'],3)
        self.assertEqual(len(result['queries']),3)
    def test_failure_records_attempt_without_retry(self):
        env=Environment();calls=[]
        def fail(action):calls.append(action);raise RuntimeError('failure')
        env.step=fail
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(RuntimeError):m.collect(env,self.plan(),out)
            self.assertTrue((Path(out)/'attempt-01.json').exists())
            self.assertTrue((Path(out)/'query-0-page-1-pre.json').exists())
        self.assertEqual(len(calls),1)
    def test_modified_schedule_rejected(self):
        plan=self.plan();plan['schedule'][0]['first_action']='click[product]';env=Environment()
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(m.Invalid):m.collect(env,plan,out)
        self.assertEqual(env.calls,[])
    def test_global_override_rejected(self):
        env=Environment();env.server.assigned_instruction_text='override'
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(m.Invalid):m.collect(env,self.plan(),out)
        self.assertEqual(env.calls,[])
if __name__=='__main__':unittest.main()
