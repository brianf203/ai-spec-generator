import unittest
from metadata import create_metadata, get_path, get_size, metadata_eq
class TestMetadata(unittest.TestCase):
    def test_create(self): m=create_metadata("/a", 100, 0); self.assertEqual(get_path(m), "/a"); self.assertEqual(get_size(m), 100)
    def test_eq(self): a=create_metadata("/a",1,0); b=create_metadata("/a",1,0); self.assertTrue(metadata_eq(a,b))