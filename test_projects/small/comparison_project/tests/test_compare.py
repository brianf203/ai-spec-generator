import unittest
from compare import min_of_three, max_of_three, clamp_range, in_range, is_sorted_asc
class TestCompare(unittest.TestCase):
    def test_min_three(self): self.assertEqual(min_of_three(3, 1, 2), 1)
    def test_max_three(self): self.assertEqual(max_of_three(1, 3, 2), 3)
    def test_clamp(self): self.assertEqual(clamp_range(15, 0, 10), 10)
    def test_in_range(self): self.assertTrue(in_range(5, 0, 10))
    def test_sorted(self): self.assertTrue(is_sorted_asc([1, 2, 3]))