import unittest
from index import add_to_index, remove_from_index, lookup_index, index_paths
from metadata import create_metadata
class TestIndex(unittest.TestCase):
    def test_add_lookup(self): idx={}; m=create_metadata("/a",1,0); add_to_index(idx,"/a",m); self.assertEqual(lookup_index(idx,"/a")["path"], "/a")
    def test_remove(self): idx={}; add_to_index(idx,"/a",{}); remove_from_index(idx,"/a"); self.assertIsNone(lookup_index(idx,"/a"))
    def test_paths(self): idx={}; add_to_index(idx,"/a",{}); add_to_index(idx,"/b",{}); self.assertEqual(len(index_paths(idx)), 2)