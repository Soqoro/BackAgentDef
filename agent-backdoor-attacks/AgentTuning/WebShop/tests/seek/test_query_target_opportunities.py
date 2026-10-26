import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('opportunities',ROOT/'scripts/seek/audit_query_target_opportunities.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class OpportunityTests(unittest.TestCase):
    def fixture(self,price=50):
        records=[];drafts=[]
        for name,ids in m.TARGETS.items():
            task=dict(id=name,price_upper=50)
            record=dict(task=task,pages=[dict(page=1,action='search[diagnostic]',available_actions={'clickables':[ids[0]]},
                products=[dict(asin=ids[0],runtime_price=price,runtime_product={'name':'candidate'})])])
            records.append(record);drafts.append(dict(task=task,source_record_hash=m.digest(record),
                proposed_benign_action='search[benign]',proposed_poison_action='search[target]'))
        return records,drafts
    def test_duplicates_and_missing_are_not_approved(self):
        records,drafts=self.fixture()
        rows=m.inspect_products([{'asin':'B007VG86Z2'},{'asin':'B007VG86Z2'}],records,drafts)
        self.assertEqual(rows[0]['candidates'][0]['catalogue_identity_status'],'ambiguous_duplicate')
        self.assertEqual(rows[0]['candidates'][1]['catalogue_identity_status'],'missing')
        self.assertFalse(rows[0]['approved_slot'])
        self.assertFalse(rows[0]['proposed_labels_executed'])
    def test_strict_price_and_unknown(self):
        for price,expected in [(50,False),(49,True),('49',None),(None,None),(True,None)]:
            records,drafts=self.fixture(price)
            rows=m.inspect_products([],records,drafts)
            self.assertIs(rows[0]['candidates'][0]['saved_page_occurrences'][0]['price_strictly_below_limit'],expected)
    def test_source_mutation_rejected(self):
        records,drafts=self.fixture();records[0]['task']['price_upper']=100
        with self.assertRaises(m.Invalid):m.inspect_products([],records,drafts)
    def test_catalogue_presence_is_not_clickability(self):
        records,drafts=self.fixture()
        rows=m.inspect_products([{'asin':'B00BVO9N2A'}],records,drafts)
        candidate=rows[0]['candidates'][1]
        self.assertEqual(candidate['catalogue_identity_status'],'unique')
        self.assertEqual(candidate['saved_page_occurrences'],[])
        self.assertFalse(candidate['task_compatibility_verified'])
if __name__=='__main__':unittest.main()
