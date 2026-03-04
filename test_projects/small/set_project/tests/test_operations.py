import unittest
from operations import union_sets, intersection_sets, is_subset, is_superset, is_disjoint
class TestOps(unittest.TestCase):
    def test_union(self): self.assertEqual(union_sets({1,2},{2,3}), {1,2,3})
    def test_intersection(self): self.assertEqual(intersection_sets({1,2},{2,3}), {2})
    def test_subset(self): self.assertTrue(is_subset({1},{1,2}))
    def test_superset(self): self.assertTrue(is_superset({1,2,3},{1,2}))
    def test_disjoint(self): self.assertTrue(is_disjoint({1,2},{3,4}))