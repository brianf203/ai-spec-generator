import unittest
from perm import factorial, perm_count, next_permutation, permute_list
class TestPerm(unittest.TestCase):
    def test_factorial(self): self.assertEqual(factorial(5), 120)
    def test_perm_count(self): self.assertEqual(perm_count(5, 2), 20)
    def test_next_perm(self): lst=[1,2,3]; next_permutation(lst); self.assertEqual(lst, [1,3,2])
    def test_permute(self): p=permute_list([1,2]); self.assertEqual(len(p), 2)