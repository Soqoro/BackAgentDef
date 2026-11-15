import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from replay_v2_sampler_accounting import control_trace
from seek.schemas import Invalid

class ControlTests(unittest.TestCase):
    def test_four_rank_remainder(self):
        r=control_trace(476)
        self.assertEqual(r['planned_updates'],118)
        self.assertEqual(r['consumed_batches_per_rank_by_epoch'],[476,468])
        self.assertEqual(r['cross_epoch_windows'],1)
    def test_divisible_epochs(self):
        r=control_trace(480)
        self.assertEqual(r['planned_updates'],120)
        self.assertEqual(r['consumed_batches_per_rank_by_epoch'],[480,480])
        self.assertEqual(r['cross_epoch_windows'],0)
    def test_short_epoch_special_case(self):
        r=control_trace(3,1)
        self.assertEqual(len(r['step_events']),1)
        self.assertEqual(len(r['step_events'][0]['microbatches']),3)
    def test_invalid(self):
        for n in (0,-1,True):
            with self.assertRaises(Invalid):control_trace(n)
    def test_no_missing_or_duplicated_consumed_positions(self):
        r=control_trace(1902)
        flat=[tuple(p) for e in r['step_events'] for p in e['microbatches']]+[tuple(p) for p in r['pending_microbatches']]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(len(flat),sum(r['consumed_batches_per_rank_by_epoch']))
