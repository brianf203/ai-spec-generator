import unittest
from linear import linear_search, contains, find_all_indices, find_first, find_last
class TestLinear(unittest.TestCase):
    def test_search(self): self.assertEqual(linear_search([1,2,3], 2), 1)
    def test_contains(self): self.assertTrue(contains([1,2,3], 2))
    def test_find_all(self): self.assertEqual(find_all_indices([1,2,1], 1), [0,2])
    def test_find_first(self): self.assertEqual(find_first([1,2,3], lambda x: x>1), 2)
    def test_find_last(self): self.assertEqual(find_last([1,2,3], lambda x: x<3), 2)