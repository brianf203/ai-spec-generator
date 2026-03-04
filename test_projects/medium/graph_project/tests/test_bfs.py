import unittest
from bfs import bfs, bfs_levels, bfs_path, bfs_bipartite
class Test(unittest.TestCase):
    def test_bfs(self): self.assertEqual(len(bfs({1:[2],2:[3],3:[]}, 1)), 3)
    def test_levels(self): self.assertEqual(bfs_levels({1:[2],2:[3],3:[]}, 1)[3], 2)
    def test_path(self): self.assertEqual(bfs_path({1:[2],2:[3],3:[]}, 1, 3), [1,2,3])
    def test_bipartite(self): self.assertTrue(bfs_bipartite({1:[2],2:[1]}))