import unittest
from path_util import basename, split_path
from index import lookup_index
class TestEdge(unittest.TestCase):
    def test_empty_path(self): self.assertEqual(basename(""), "")
    def test_split_empty(self): self.assertEqual(split_path(""), [])
    def test_lookup_missing(self): self.assertIsNone(lookup_index({},"/x"))