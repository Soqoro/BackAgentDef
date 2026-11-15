import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from propose_v2_exact_passes import schedule,verify
from seek.schemas import Invalid

class ExactPassTests(unittest.TestCase):
    def test_all_topologies(self):
        for world,updates in ((1,476),(2,238),(4,120),(8,60)):
            r=verify(schedule(1902,world),1902,world)
            self.assertEqual(r['updates'],updates);self.assertEqual(r['valid_presentations'],3804)
            self.assertEqual(r['records_with_exactly_two_exposures'],1902)
    def test_repeatable_and_seed_specific(self):
        self.assertEqual(schedule(19,4),schedule(19,4))
        self.assertNotEqual(schedule(19,4),schedule(19,4,seed=5))
    def test_small_dataset_and_padding(self):
        r=verify(schedule(1,8),1,8)
        self.assertEqual(r['empty_rank_slots'],14);self.assertEqual(r['valid_presentations'],2)
    def test_missing_and_duplicate_rejected(self):
        x=schedule(10,2)
        with self.assertRaises(Invalid):verify(x[:-1],10,2)
        x[0]['microsteps'][0][1]=x[0]['microsteps'][0][0]
        with self.assertRaises(Invalid):verify(x,10,2)
    def test_divisible_no_padding(self):
        r=verify(schedule(32,4),32,4)
        self.assertEqual(r['empty_rank_slots'],0);self.assertEqual(r['partial_windows'],0)
