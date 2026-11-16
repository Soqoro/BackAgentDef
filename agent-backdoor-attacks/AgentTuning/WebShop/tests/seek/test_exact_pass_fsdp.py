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
