import sys,unittest
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from check_v2_loss_scaling import check_window,example_gradient,token_values
from propose_v2_exact_passes import schedule
from seek.schemas import Invalid

class LossTests(unittest.TestCase):
    def test_full_and_partial_windows(self):
        for world in (1,2,4,8):
            for w in schedule(19,world):
                r=check_window(w,world);self.assertEqual(r['reference'],r['reduced'])
    def test_empty_rank_participates(self):
        w=schedule(1,8,epochs=1)[0];r=check_window(w,8)
        self.assertEqual(len(r['trace']),8);self.assertEqual(sum(t[2]=='zero' for t in r['trace']),7)
    def test_extra_accumulation_division_is_wrong(self):
        r=check_window(schedule(19,4)[0],4)
        self.assertNotEqual(r['reference'],r['reduced']/8)
    def test_token_weighting_is_a_different_objective(self):
        a=(example_gradient(0)+example_gradient(1))/2
        b=(example_gradient(0)*len(token_values(0))+example_gradient(1)*len(token_values(1)))/(len(token_values(0))+len(token_values(1)))
        self.assertNotEqual(a,b)
    def test_incorrect_count_rejected(self):
        w=schedule(1,8)[0];w['valid_records']=8
        with self.assertRaises(Invalid):check_window(w,8)
