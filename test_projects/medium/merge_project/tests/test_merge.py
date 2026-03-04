import unittest
from merge import merge_sorted_lists, merge_k_sorted, merge_with_duplicates, merge_intervals, merge_count_inversions
class Test(unittest.TestCase):
    def test_merge(self): self.assertEqual(merge_sorted_lists([1,3],[2,4]), [1,2,3,4])
    def test_k_merge(self): self.assertEqual(merge_k_sorted([[1,4],[2,3]]), [1,2,3,4])
    def test_dup(self): self.assertEqual(merge_with_duplicates([1,2],[2,3]), [1,2,3])
    def test_intervals(self): self.assertEqual(merge_intervals([[1,3],[2,4]]), [[1,4]])
    def test_inversions(self): self.assertEqual(merge_count_inversions([2,1]), 1)