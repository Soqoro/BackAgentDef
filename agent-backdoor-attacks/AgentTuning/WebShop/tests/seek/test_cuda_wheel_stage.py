import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from stage_cuda_environment_wheels import candidates,run,PINS

class StageTests(unittest.TestCase):
    def report(self):
        return {'install':[{'metadata':{'name':n,'version':v},'download_info':{
            'url':('https://download.pytorch.org/' if n=='torch' else 'https://files.pythonhosted.org/')+n+'.whl',
            'archive_info':{'hashes':{'sha256':'a'*64}}}} for n,v in PINS.items()]}
    def test_no_authorization_before_any_io(self):
        with self.assertRaisesRegex(ValueError,'authorization'):run('/never-create-this',False)
    def test_pins_accepted(self):self.assertEqual(len(candidates(self.report())),5)
    def test_changed_torch_rejected(self):
        r=self.report();r['install'][0]['metadata']['version']='2.6.0+cpu'
        with self.assertRaisesRegex(ValueError,'pin'):candidates(r)
    def test_torch_wrong_origin(self):
        r=self.report();r['install'][0]['download_info']['url']='https://files.pythonhosted.org/torch.whl'
        with self.assertRaisesRegex(ValueError,'official'):candidates(r)
    def test_duplicate_and_hashless_rejected(self):
        r=self.report();r['install'].append(r['install'][0])
        with self.assertRaises(ValueError):candidates(r)
        r=self.report();r['install'][0]['download_info']['archive_info']={}
        with self.assertRaises(ValueError):candidates(r)
