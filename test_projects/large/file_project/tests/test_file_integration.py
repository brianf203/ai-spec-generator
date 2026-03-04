import unittest
from path_util import join_path, basename
from metadata import create_metadata
from index import add_to_index, lookup_index
class TestIntegration(unittest.TestCase):
    def test_full_flow(self): idx={}; p=join_path("dir","file.txt"); m=create_metadata(p,10,0); add_to_index(idx,p,m); self.assertEqual(lookup_index(idx,p)["path"], p)