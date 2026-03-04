import unittest
from search import binary_search, binary_search_left, binary_search_right, binary_search_closest, is_in_sorted
class Test(unittest.TestCase):
    def test_bs(self): self.assertEqual(binary_search([1,2,3,4], 3), 2)
    def test_left(self): self.assertEqual(binary_search_left([1,2,2,3], 2), 1)
    def test_right(self): self.assertEqual(binary_search_right([1,2,2,3], 2), 2)
    def test_closest(self): self.assertIn(binary_search_closest([1,3,5], 4), [1,2])
    def test_in(self): self.assertTrue(is_in_sorted([1,2,3], 2))