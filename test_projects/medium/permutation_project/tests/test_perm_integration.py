import unittest
from perm import permute_list, factorial
from comb import combine_list, comb_count
class TestIntegration(unittest.TestCase):
    def test_perm_len(self): lst=[1,2,3]; self.assertEqual(len(permute_list(lst)), factorial(len(lst)))
    def test_comb_count(self): lst=[1,2,3,4,5]; self.assertEqual(len(combine_list(lst, 2)), comb_count(5, 2))