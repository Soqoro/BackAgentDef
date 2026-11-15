import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from prepare_v2_training_spec import proposed_recipe
from seek.schemas import Invalid

class TrainingSpecTests(unittest.TestCase):
    def parent(self):
        return {'recipe':dict(optimizer=None,gpus=None,epochs=2,batch_per_gpu=1,
            gradient_accumulation=8,seed=271028,checkpoint_selection='fixed_final_epoch')}
    def test_preserves_bound_recipe_and_unknowns(self):
        p=self.parent();r=proposed_recipe(p)
        self.assertEqual(r['inherited'],p['recipe'])
        for k in ('expected_updates','optimizer_selection','topology','loss_normalization','initialization_weight_manifest'):
            self.assertIsNone(r[k])
        self.assertEqual(r['status'],'unresolved_not_launchable')
    def test_does_not_mutate_parent(self):
        p=self.parent();r=proposed_recipe(p);r['inherited']['seed']=1
        self.assertEqual(p['recipe']['seed'],271028)
    def test_no_silent_optimizer_or_topology_resolution(self):
        for key,value in (('optimizer','adamw_torch'),('gpus',4)):
            p=self.parent();p['recipe'][key]=value
            with self.assertRaises(Invalid):proposed_recipe(p)
