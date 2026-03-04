import unittest
from operations import flatten_list, rotate_list, reverse_list, first_n, last_n
class TestOpsEdge(unittest.TestCase):
    def test_flatten_empty(self): self.assertEqual(flatten_list([]), [])
    def test_rotate_empty(self): self.assertEqual(rotate_list([], 1), [])
    def test_rotate_full(self): self.assertEqual(rotate_list([1,2,3], 3), [1,2,3])
    def test_last_n_zero(self): self.assertEqual(last_n([1,2,3], 0), [])
    def test_first_n_excess(self): self.assertEqual(first_n([1,2], 5), [1,2])