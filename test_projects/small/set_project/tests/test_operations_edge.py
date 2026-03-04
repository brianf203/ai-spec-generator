import unittest
from operations import difference_sets, symmetric_difference, is_subset, is_disjoint
class TestOpsEdge(unittest.TestCase):
    def test_difference(self): self.assertEqual(difference_sets({1,2,3},{2}), {1,3})
    def test_symmetric_diff(self): self.assertEqual(symmetric_difference({1,2},{2,3}), {1,3})
    def test_subset_empty(self): self.assertTrue(is_subset(set(), {1,2}))
    def test_disjoint_empty(self): self.assertTrue(is_disjoint(set(), {1,2}))