"""Additional set operation tests."""
import unittest
from operations import union_sets, intersection_sets, is_subset, is_superset


class TestSetExtra(unittest.TestCase):
    def test_union_empty(self):
        self.assertEqual(union_sets(set(), {1, 2}), {1, 2})

    def test_intersection_disjoint(self):
        self.assertEqual(intersection_sets({1, 2}, {3, 4}), set())

    def test_subset_empty(self):
        self.assertTrue(is_subset(set(), {1, 2, 3}))

    def test_superset_equal(self):
        self.assertTrue(is_superset({1, 2}, {1, 2}))
