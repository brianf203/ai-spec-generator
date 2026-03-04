import unittest
from operations import flatten_list, rotate_list, reverse_list, first_n, last_n
class TestOps(unittest.TestCase):
    def test_flatten(self): self.assertEqual(flatten_list([1,[2,3]]), [1,2,3])
    def test_rotate(self): self.assertEqual(rotate_list([1,2,3], 1), [3,1,2])
    def test_reverse(self): self.assertEqual(reverse_list([1,2,3]), [3,2,1])
    def test_first_n(self): self.assertEqual(first_n([1,2,3,4], 2), [1,2])
    def test_last_n(self): self.assertEqual(last_n([1,2,3,4], 2), [3,4])