import unittest
from merge import merge_sorted_lists, merge_intervals, merge_count_inversions
class TestEdge(unittest.TestCase):
    def test_merge_empty(self): self.assertEqual(merge_sorted_lists([], [1,2]), [1,2])
    def test_merge_both_empty(self): self.assertEqual(merge_sorted_lists([], []), [])
    def test_intervals_empty(self): self.assertEqual(merge_intervals([]), [])
    def test_inversions_empty(self): self.assertEqual(merge_count_inversions([]), 0)