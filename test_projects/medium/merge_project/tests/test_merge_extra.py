"""Additional merge sort tests."""
import unittest
from mergesort import mergesort
from merge import merge_sorted_lists


class TestMergeExtra(unittest.TestCase):
    def test_mergesort_single(self):
        self.assertEqual(mergesort([5]), [5])

    def test_mergesort_two(self):
        self.assertEqual(mergesort([2, 1]), [1, 2])

    def test_merge_sorted_halves(self):
        self.assertEqual(merge_sorted_lists([1, 3], [2, 4]), [1, 2, 3, 4])
