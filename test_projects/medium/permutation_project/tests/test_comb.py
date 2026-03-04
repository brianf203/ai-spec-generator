import unittest
from comb import comb_count, combine_list
class TestComb(unittest.TestCase):
    def test_comb_count(self): self.assertEqual(comb_count(5, 2), 10)
    def test_combine(self): c=combine_list([1,2,3], 2); self.assertEqual(len(c), 3)