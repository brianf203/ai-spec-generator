"""Additional graph tests."""
import unittest
from bfs import bfs, bfs_levels
from dfs import dfs


class TestGraphExtra(unittest.TestCase):
    def test_bfs_single_node(self):
        self.assertEqual(bfs({1: []}, 1), [1])

    def test_bfs_levels_single(self):
        self.assertEqual(bfs_levels({1: []}, 1), {1: 0})

    def test_dfs_single_node(self):
        self.assertEqual(dfs({1: []}, 1), [1])
