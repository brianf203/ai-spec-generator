import unittest
from search import binary_search, binary_search_left, binary_search_right, is_in_sorted
from rotated import min_in_rotated
class TestEdge(unittest.TestCase):
    def test_not_found(self): self.assertEqual(binary_search([1,2,3], 5), -1)
    def test_empty(self): self.assertEqual(binary_search([], 1), -1)
    def test_left_not_found(self): self.assertEqual(binary_search_left([1,3,5], 2), -1)
    def test_is_in_false(self): self.assertFalse(is_in_sorted([1,2,3], 5))
    def test_min_empty(self): self.assertIsNone(min_in_rotated([]))