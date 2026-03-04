"""Additional quicksort tests."""
import unittest
from sort import quicksort_copy


class TestQuicksortExtra(unittest.TestCase):
    def test_quicksort_single(self):
        self.assertEqual(quicksort_copy([5]), [5])

    def test_quicksort_empty(self):
        self.assertEqual(quicksort_copy([]), [])

    def test_quicksort_duplicates(self):
        self.assertEqual(quicksort_copy([2, 1, 2, 3, 1]), [1, 1, 2, 2, 3])
