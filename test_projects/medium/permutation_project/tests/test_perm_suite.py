import unittest
from perm import perm_count, next_permutation
from comb import comb_count
class TestSuite(unittest.TestCase):
    def test_perm_count(self): self.assertEqual(perm_count(5, 2), 20)
    def test_comb(self): self.assertEqual(comb_count(5, 2), 10)