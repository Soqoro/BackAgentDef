import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from check_v2_cpu_collectives import rank_window
from propose_v2_exact_passes import schedule
from seek.schemas import Invalid

class RankWindowTests(unittest.TestCase):
    def test_rank_partition_has_no_duplicate_real_examples(self):
        for world in (2,4):
            for window in schedule(1902,world):
                locals=[rank_window(window,r,world)[0] for r in range(world)]
                self.assertEqual(len({len(x) for x in locals}),1)
                ids=[i for x in locals for i in x if i is not None]
                self.assertEqual(len(ids),len(set(ids)))
                self.assertEqual(len(ids),window['valid_records'])
    def test_empty_rank_stress(self):
        w=schedule(1,4,epochs=1)[0]
        self.assertEqual(sum(rank_window(w,r,4)[0]==[None] for r in range(4)),3)
    def test_invalid_rank_and_width(self):
        w=schedule(1,4)[0]
        with self.assertRaises(Invalid):rank_window(w,4,4)
        with self.assertRaises(Invalid):rank_window(w,0,2)
    def test_bad_count(self):
        w=schedule(1,4)[0];w['valid_records']=4
        with self.assertRaises(Invalid):rank_window(w,0,4)
