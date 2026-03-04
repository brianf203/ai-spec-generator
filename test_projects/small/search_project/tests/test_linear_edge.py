import unittest
from linear import linear_search, contains, find_all_indices, find_first, find_last
class TestLinearEdge(unittest.TestCase):
    def test_search_not_found(self): self.assertEqual(linear_search([1,2,3], 5), -1)
    def test_search_empty(self): self.assertEqual(linear_search([], 1), -1)
    def test_contains_false(self): self.assertFalse(contains([1,2,3], 5))
    def test_find_first_none(self): self.assertIsNone(find_first([1,2,3], lambda x: x>10))
    def test_find_last_none(self): self.assertIsNone(find_last([], lambda x: True))