import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'scripts/seek'))
from exact_pass_runtime import BoundWindows
from seek.schemas import digest, Invalid


class BoundTests(unittest.TestCase):
    def make(self, world=4):
        rows=[{'id':str(i)} for i in range(19)]; annotations=[{'loss':True}]*19
        return rows, BoundWindows(rows, annotations, digest(rows), digest(annotations), world)

    def test_actual_ids_once_each_pass(self):
        rows, bound=self.make()
        from collections import Counter
        seen=Counter()
        for rank in range(4):
            for window in bound.rank_batches(rank):
                for slot in window['slots']:
                    if slot is not None:
                        row=bound.fetch(rows, slot); seen[window['epoch'],row['id']]+=1
        self.assertEqual(seen, Counter({(e,str(i)):1 for e in range(2) for i in range(19)}))

    def test_changed_dataset_order_rejected(self):
        rows,bound=self.make(); slot=next(s for w in bound.rank_batches(0) for s in w['slots'] if s is not None)
        rows[slot[0]]={'id':'wrong'}
        with self.assertRaisesRegex(Invalid,'dataset_order_changed'):bound.fetch(rows,slot)

    def test_corpus_and_annotation_tampering_rejected(self):
        with self.assertRaises(Invalid):BoundWindows([{'id':'x'}], [], 'wrong',digest([]),1)
        with self.assertRaises(Invalid):BoundWindows([{'id':'x'}], [],digest([{'id':'x'}]),'wrong',1)

    def test_duplicate_ids_rejected(self):
        rows=[{'id':'x'}]*2
        with self.assertRaisesRegex(Invalid,'unique_record_ids'):BoundWindows(rows,[],digest(rows),digest([]),1)

    def test_empty_slot_cannot_fetch_real_data(self):
        rows,bound=self.make()
        with self.assertRaises(Invalid):bound.fetch(rows,None)

    def test_invalid_rank(self):
        _,bound=self.make()
        with self.assertRaises(Invalid):list(bound.rank_batches(4))
