"""Only new delivery-boundary checks; no model or candidate redesign."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
from verify_csham_review_packet_v3 import archive,resolve

class PacketTests(unittest.TestCase):
    def make(self,name='data.json',raw=b'{}',expected=None,duplicate=False,symlink=False):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            info=zipfile.ZipInfo(name)
            if symlink:info.external_attr=0o120777<<16
            z.writestr(info,raw)
            if duplicate:z.writestr(name,raw)
            z.writestr('manifest.json',json.dumps({name:dict(bytes=len(raw),sha256=expected or hashlib.sha256(raw).hexdigest())}))
        return stream.getvalue()
    def test_good(self):self.assertEqual(archive(self.make()).read('data.json'),b'{}')
    def test_tampering(self):
        with self.assertRaisesRegex(ValueError,'digest'):archive(self.make(expected='0'*64))
    def test_path_escape(self):
        for name in ('../secret','/tmp/secret','a\\b','C:/x'):
            with self.assertRaisesRegex(ValueError,'unsafe'):archive(self.make(name=name))
    def test_duplicate(self):
        with self.assertWarns(UserWarning):raw=self.make(duplicate=True)
        with self.assertRaisesRegex(ValueError,'duplicate'):archive(raw)
    def test_symlink(self):
        with self.assertRaisesRegex(ValueError,'symlink'):archive(self.make(symlink=True))
    def test_nested_resolution(self):
        child=self.make();z=archive(self.make(name='child.zip',raw=child))
        self.assertEqual(resolve(z,'child.zip!data.json',{}),b'{}')

if __name__=='__main__':unittest.main()
