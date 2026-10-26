import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import check_fragrance_queries as m
from test_constructed_collection import Environment

class FragranceTests(unittest.TestCase):
    def plan(self):return dict(hash='fixture',actions=list(m.ACTIONS),maximum_environment_actions=2,task={'instruction':'Find fragrance'})
    def test_exact_two_actions_fresh_sessions_no_pagination(self):
        env=Environment();original=copy.deepcopy(env.server.goals)
        with tempfile.TemporaryDirectory() as out:
            rows=m.collect(env,self.plan(),out)
        self.assertEqual(env.calls,list(m.ACTIONS))
        self.assertNotEqual(rows[0]['session'],rows[1]['session'])
        self.assertEqual(env.server.goals,original)
        self.assertTrue(all(r['status']=='fixed_target_not_visible' for r in rows))
    def test_failure_no_retry_and_pre_record_saved(self):
        env=Environment();calls=[]
        def fail(action):calls.append(action);raise RuntimeError('failure')
        env.step=fail
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(RuntimeError):m.collect(env,self.plan(),out)
            self.assertTrue((Path(out)/'pre-0.json').is_file())
        self.assertEqual(calls,[m.ACTIONS[0]])
    def test_modified_query_rejected(self):
        plan=self.plan();plan['actions'][0]='search[other]';env=Environment()
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(m.Invalid):m.collect(env,plan,out)
        self.assertEqual(env.calls,[])
    def test_visible_target_does_not_establish_compatibility(self):
        env=Environment();env.server.product_item_dict['B007VG86Z2']={'name':'candidate'}
        env.server.product_prices['B007VG86Z2']=50
        old=env.get_available_actions
        def actions():
            result=old()
            if env.page:result['clickables'].append('b007vg86z2')
            return result
        env.get_available_actions=actions
        with tempfile.TemporaryDirectory() as out:rows=m.collect(env,self.plan(),out)
        target=next(p for p in rows[0]['products'] if p['fixed_target_candidate'])
        self.assertFalse(target['price_below_50'])
        self.assertFalse(target['task_compatibility_verified'])
if __name__=='__main__':unittest.main()
