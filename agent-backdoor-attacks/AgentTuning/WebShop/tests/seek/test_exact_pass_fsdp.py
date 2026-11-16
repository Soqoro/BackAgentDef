import sys,unittest
from fractions import Fraction
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from exact_pass_fsdp import recipe,windows_for_role,backward_window,decay_names,optimizer_groups,execute
from seek.schemas import digest,Invalid

class FsdpTests(unittest.TestCase):
    def test_real_exposures_carriers_and_updates(self):
        records=[{'id':str(i)} for i in range(1902)];seen=Counter();carriers=0
        for rank in range(4):
            bound,windows=windows_for_role(records,[],digest(records),digest([]),rank)
            self.assertEqual(len(windows),238)
            for w in windows:
                for s in w['slots']:
                    if s is None:
                        carriers+=1;bound.fetch(records,w['carrier'])
                    else:seen[w['epoch'],s[1]]+=1
        self.assertEqual(carriers,4)
        self.assertEqual(seen,Counter({(e,str(i)):1 for e in range(2) for i in range(1902)}))
    def test_no_no_sync_and_zero_weight(self):
        gradients=[];calls=[]
        class Loss:
            ndim=0
            def __init__(self,v):self.v=v
            def __mul__(self,v):return Loss(self.v*v)
            def __truediv__(self,v):return Loss(self.v/v)
            def backward(self):gradients.append(self.v)
        class Wrapper:
            def no_sync(self):raise AssertionError('FSDP no_sync forbidden')
        w={'slots':[(0,'a'),None],'carrier':(0,'a'),'valid_records':6}
        def forward(model,s):calls.append(s);return Loss(Fraction(3))
        backward_window(Wrapper(),w,forward)
        self.assertEqual(calls,[(0,'a')]*2);self.assertEqual(gradients,[2,0])
    def test_pre_wrap_group_names_survive_sharded_shapes(self):
        class P:
            requires_grad=True
            def __init__(self,n):self.ndim=n
        matrix=P(2);norm=P(1);pairs=[('weight',matrix),('norm.weight',norm)]
        decay=decay_names(pairs);matrix.ndim=1
        self.assertEqual(optimizer_groups(pairs,decay)[0]['params'],[matrix])
    def test_gate_blocks_before_build(self):
        with self.assertRaises(Invalid):
            execute({}, {}, lambda:{},lambda:self.fail('model loaded'),lambda x:None)
    def test_recipe_preserves_global_window(self):
        r=recipe();self.assertEqual(r['world_size']*r['accumulation_max'],16)
        self.assertFalse(r['recipe_approved']);self.assertFalse(r['cpu_offload'])

class ThreeRankFsdpTests(unittest.TestCase):
    def test_exact_membership_partial_and_no_carriers(self):
        records=[{'id':str(i)} for i in range(1902)]; seen=Counter()
        for rank in range(3):
            bound,windows=windows_for_role(records,[],digest(records),digest([]),rank,3)
            self.assertEqual(len(windows),254)
            self.assertEqual([w['valid_records'] for w in windows if len(w['slots'])<5],[12,12])
            self.assertEqual(sum(len(w['slots']) for w in windows),1268)
            for w in windows:
                self.assertNotIn('carrier',w)
                for slot in w['slots']:
                    self.assertIsNotNone(slot);seen[w['epoch'],bound.fetch(records,slot)['id']]+=1
        self.assertEqual(seen,Counter({(e,str(i)):1 for e in range(2) for i in range(1902)}))
    def test_partial_scaling(self):
        gradients=[]
        class Loss:
            ndim=0
            def __init__(self,v):self.v=v
            def __mul__(self,v):return Loss(self.v*v)
            def __truediv__(self,v):return Loss(self.v/v)
            def backward(self):gradients.append(self.v)
        backward_window(object(),dict(slots=[(i,str(i)) for i in range(4)],valid_records=12),
                        lambda m,s:Loss(Fraction(1)),3)
        self.assertEqual(gradients,[Fraction(1,4)]*4)
    def test_bad_tail_fails_before_forward(self):
        for slots,count in (([None],3),([(0,'a')],2)):
            with self.assertRaises(Invalid):
                backward_window(object(),dict(slots=slots,valid_records=count),
                                lambda m,s:self.fail('forward called'),3)
    def test_recipe_revision(self):
        from exact_pass_training import learning_rate
        from exact_pass_fsdp import check_recipe
        r=recipe(3); check_recipe(r)
        self.assertEqual((r['accumulation_max'],r['updates'],r['warmup_updates']),(5,254,6))
        self.assertEqual(r['checkpoint'],'one_final_after_update_254')
        self.assertAlmostEqual(learning_rate(6,r),5e-5)
        self.assertGreater(learning_rate(254,r),0)
        r['updates']=238
        with self.assertRaises(Invalid):check_recipe(r)
    def test_unsupported_topology(self):
        for world in (1,2,5,True):
            with self.assertRaises(Invalid):recipe(world)
