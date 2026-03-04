import unittest
from compare import clamp_range, is_sorted_asc, is_sorted_desc
class TestEdge(unittest.TestCase):
    def test_clamp_inside(self): self.assertEqual(clamp_range(5, 0, 10), 5)
    def test_sorted_empty(self): self.assertTrue(is_sorted_asc([]))
    def test_sorted_desc(self): self.assertTrue(is_sorted_desc([3, 2, 1]))