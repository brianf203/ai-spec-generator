import unittest
from search import index_of, count_val, contains, min_idx, max_idx
class TestSearch(unittest.TestCase):
    def test_index_of(self): self.assertEqual(index_of([1,2,3], 2), 1)
    def test_count_val(self): self.assertEqual(count_val([1,2,1], 1), 2)
    def test_contains(self): self.assertTrue(contains([1,2,3], 2))
    def test_min_idx(self): self.assertEqual(min_idx([3,1,2]), 1)
    def test_max_idx(self): self.assertEqual(max_idx([1,3,2]), 1)