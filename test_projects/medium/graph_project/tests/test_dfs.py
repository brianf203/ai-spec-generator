import unittest
from dfs import dfs, dfs_iterative, dfs_path, dfs_cycle, topological_sort, count_components
class Test(unittest.TestCase):
    def test_dfs(self): self.assertEqual(len(dfs({1:[2],2:[3],3:[]}, 1)), 3)
    def test_iter(self): self.assertEqual(len(dfs_iterative({1:[2],2:[3],3:[]}, 1)), 3)
    def test_path(self): self.assertEqual(dfs_path({1:[2],2:[3],3:[]}, 1, 3), [1,2,3])
    def test_cycle(self): self.assertTrue(dfs_cycle({1:[2],2:[1]}))
    def test_topo(self): self.assertEqual(len(topological_sort({1:[2],2:[3],3:[]})), 3)
    def test_components(self): self.assertEqual(count_components({1:[2],2:[1],3:[]}), 2)