import unittest
from path_util import join_path, split_path, basename, dirname
class TestPath(unittest.TestCase):
    def test_join(self): self.assertEqual(join_path("a","b","c"), "a/b/c")
    def test_split(self): self.assertEqual(split_path("a/b/c"), ["a","b","c"])
    def test_basename(self): self.assertEqual(basename("a/b/c"), "c")
    def test_dirname(self): self.assertEqual(dirname("a/b/c"), "a/b")