import unittest
from bubble import bubble_sort, is_sorted
from selection import min_index, max_index
from insertion import insert_sorted
class TestEdge(unittest.TestCase):
    def test_empty(self): self.assertEqual(bubble_sort([]), [])
    def test_single(self): self.assertEqual(bubble_sort([1]), [1])
    def test_is_sorted_empty(self): self.assertTrue(is_sorted([]))
    def test_min_idx_empty(self): self.assertEqual(min_index([], 0), -1)
    def test_insert_empty(self): self.assertEqual(insert_sorted([], 5), [5])