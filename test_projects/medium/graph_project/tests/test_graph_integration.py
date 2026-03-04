import unittest
from bfs import bfs, bfs_path
from dfs import dfs, dfs_path
class TestIntegration(unittest.TestCase):
    def test_bfs_dfs_same_nodes(self): g={1:[2,3],2:[],3:[]}; self.assertEqual(sorted(bfs(g,1)), sorted(dfs(g,1)))
    def test_path_consistency(self): g={1:[2],2:[3],3:[]}; self.assertEqual(bfs_path(g,1,3), dfs_path(g,1,3))