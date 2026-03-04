import unittest
from perm import next_permutation, permute_list
class TestEdge(unittest.TestCase):
    def test_next_last(self): lst=[3,2,1]; self.assertFalse(next_permutation(lst))
    def test_permute_empty(self): self.assertEqual(permute_list([]), [[]])