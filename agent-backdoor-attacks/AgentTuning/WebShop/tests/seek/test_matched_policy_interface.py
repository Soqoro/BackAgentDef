import unittest
from seek.three_way.policy_interface import render,prefix_check,CONTRACT
from seek.schemas import Invalid

class Conv:
    roles=('USER','ASSISTANT')
    def append_message(self,role,text):self.messages.append((role,text))
    def get_prompt(self):return 'default system\n'+'\n'.join(r+':'+(t or '') for r,t in self.messages)

class InterfaceTests(unittest.TestCase):
    def test_shared_template_and_system(self):
        names=[]
        def make(name):names.append(name);return Conv()
        turns=[dict(from_='human',value='request'),dict(from_='gpt',value='action')]
        turns=[{'from':x['from_'],'value':x['value']} for x in turns]
        generation=render(make,turns[:-1],True)
        training=render(make,turns)
        self.assertEqual(training,generation+'action')
        self.assertEqual(names,['vicuna','vicuna'])
        self.assertFalse(CONTRACT['installed'])
        self.assertFalse(CONTRACT['training_authorized'])
    def test_prefix_boundary_not_similar_text(self):
        self.assertTrue(prefix_check([1,2],[1,2,3],2))
        self.assertFalse(prefix_check([1,9],[1,2,3],2))
        self.assertFalse(prefix_check([1,2,3],[1,2,3],2))
        self.assertFalse(prefix_check([],[],0))
    def test_bad_roles_and_parity(self):
        with self.assertRaises(Invalid):render(lambda _:Conv(),[{'from':'gpt','value':'x'}],True)
        with self.assertRaises(Invalid):render(lambda _:Conv(),[{'from':'human','value':'x'}])

if __name__=='__main__':unittest.main()
