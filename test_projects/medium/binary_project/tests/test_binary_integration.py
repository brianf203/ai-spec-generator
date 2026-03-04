import unittest
from search import binary_search, binary_search_range
from recursive import binary_search_recursive
from rotated import search_rotated, find_rotation_point
class TestIntegration(unittest.TestCase):
    def test_bs_eq_recursive(self): lst=[1,2,3,4,5]; self.assertEqual(binary_search(lst,3), binary_search_recursive(lst,3))
    def test_range(self): lst=[1,2,2,2,3]; lo,hi=binary_search_range(lst,2); self.assertEqual(lo, 1); self.assertEqual(hi, 3)
    def test_rotated_search(self): lst=[4,5,1,2,3]; self.assertEqual(search_rotated(lst, 5), 1)