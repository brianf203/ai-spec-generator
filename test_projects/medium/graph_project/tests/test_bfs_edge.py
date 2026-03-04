import unittest
from bfs import bfs, bfs_path, bfs_bipartite
class TestEdge(unittest.TestCase):
    def test_bfs_single(self): self.assertEqual(bfs({1:[]}, 1), [1])
    def test_path_same(self): self.assertEqual(bfs_path({1:[2]}, 1, 1), [1])
    def test_path_unreachable(self): self.assertEqual(bfs_path({1:[2],2:[]}, 1, 3), [])
    def test_bipartite_single(self): self.assertTrue(bfs_bipartite({1:[]}))