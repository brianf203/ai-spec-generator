import unittest
from list_ops import length, to_list, from_list, append, reverse
class TestListOps(unittest.TestCase):
    def test_length(self): h=from_list([1,2,3]); self.assertEqual(length(h), 3)
    def test_to_list(self): h=from_list([1,2]); self.assertEqual(to_list(h), [1,2])
    def test_append(self): h=from_list([1]); h=append(h,2); self.assertEqual(to_list(h), [1,2])
    def test_reverse(self): h=from_list([1,2,3]); self.assertEqual(to_list(reverse(h)), [3,2,1])