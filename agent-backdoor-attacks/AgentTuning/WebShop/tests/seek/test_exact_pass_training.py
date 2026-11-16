import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from exact_pass_training import recipe, learning_rate, update_loop, orchestrate, parameter_groups
from exact_pass_runtime import BoundWindows
from seek.schemas import digest, Invalid


class TrainingTests(unittest.TestCase):
    def windows(self):
        rows=[{'id':str(i)} for i in range(1902)]
        return list(BoundWindows(rows,[],digest(rows),digest([]),2).rank_batches(0))
    def test_update_sequence_and_checkpoint(self):
        events=[]
        update_loop(self.windows(),recipe(),lambda:events.append('zero'),lambda w:events.append('backward'),
            lambda:True,lambda v:events.append('clip'),lambda lr:events.append('step'),lambda:events.append('save'))
        self.assertEqual(events, ['zero','backward','clip','step']*238+['save'])
    def test_nonfinite_aborts_without_step_or_checkpoint(self):
        events=[]
        with self.assertRaisesRegex(Invalid,'nonfinite_gradient'):
            update_loop(self.windows(),recipe(),lambda:None,lambda w:None,lambda:False,
                lambda v:events.append('clip'),lambda lr:events.append('step'),lambda:events.append('save'))
        self.assertEqual(events,[])
    def test_empty_rejected_before_any_update(self):
        windows=self.windows(); windows[-1]['slots']=(None,)
        with self.assertRaisesRegex(Invalid,'empty_slots'):
            update_loop(windows,recipe(),lambda:self.fail('side effect'),None,None,None,None,None)
    def test_learning_rate(self):
        s=recipe(); rates=[learning_rate(i,s) for i in range(1,239)]
        self.assertAlmostEqual(rates[0],1e-5)
        self.assertEqual(rates[4],5e-5)
        self.assertTrue(all(a>=b for a,b in zip(rates[5:],rates[6:])))
        self.assertGreater(rates[-1],0)
    def test_absent_grant_stops_before_build(self):
        with self.assertRaises(Invalid):orchestrate({}, {}, {}, lambda:self.fail('loaded'),lambda x:None)
    def test_binding_mismatch_stops_before_build(self):
        keys=('training_authorized','installation_authorized','resource_approved','recipe_approved','final_corpus_approved')
        plan=dict(schema='seek-exact-pass-launch-v1',bindings={k:'bound' for k in ('code','corpus','annotations','native_contract','initialization','schedule','recipe','topology','resources','output')},**dict.fromkeys(keys,True))
        grant=dict(plan_hash=digest(plan),**dict.fromkeys(keys,True))
        with self.assertRaisesRegex(Invalid,'binding_mismatch'):
            orchestrate(plan,grant,{},lambda:self.fail('loaded'),lambda x:None)
    def test_decay_partition_and_duplicate_rejection(self):
        class P:
            requires_grad=True
            def __init__(self,n):self.ndim=n
        matrix=P(2); bias=P(1)
        groups=parameter_groups([('weight',matrix),('bias',bias)],recipe())
        self.assertEqual(groups[0]['params'],[matrix]);self.assertEqual(groups[1]['params'],[bias])
        with self.assertRaises(Invalid):parameter_groups([('a',matrix),('b',matrix)],recipe())
